import cv2
import numpy as np
import time
import threading
import os
import ctypes

from .database import get_all_face_samples, log_access
from .alerts import AlertManager

# ── Detección real de GPU ─────────────────────────────────────────────────────
# get_available_providers() solo lista providers compilados, NO verifica DLLs.
# Probamos cargar cublasLt64_12.dll (CUDA 12 runtime) para confirmarlo.
_CUDA_AVAILABLE = False
_GPU_NAME       = "CPU"

try:
    import onnxruntime as _ort
    if "CUDAExecutionProvider" in _ort.get_available_providers():
        try:
            ctypes.CDLL("cublas64_12.dll")
            ctypes.CDLL("cublasLt64_12.dll")
            _CUDA_AVAILABLE = True
            try:
                import subprocess
                _GPU_NAME = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                    timeout=3
                ).decode().strip().splitlines()[0]
            except Exception:
                _GPU_NAME = "NVIDIA GPU"
        except OSError:
            print("[OmniFace] CUDA no disponible: falta CUDA 12 runtime (cublasLt64_12.dll)")
            print("[OmniFace] → Instala CUDA Toolkit 12.x desde developer.nvidia.com/cuda-downloads")
except ImportError:
    pass

_CPU_CORES = os.cpu_count() or 2

# ── Parámetros adaptativos ────────────────────────────────────────────────────
if _CUDA_AVAILABLE:
    _MODEL_NAME     = 'buffalo_l'    # ResNet50 — GPU
    _DET_SIZE       = (640, 480)
    PROCESS_EVERY_N = 1
    print(f"[OmniFace] GPU: {_GPU_NAME} — InsightFace buffalo_l @ 640×480")
else:
    _MODEL_NAME     = 'buffalo_sc'   # MobileFaceNet — CPU
    _DET_SIZE       = (256, 192)     # 4:3, ambas dims múltiplo de 32 (stride RetinaFace)
    PROCESS_EVERY_N = 3              # 1 de cada 3 frames
    print(f"[OmniFace] CPU ({_CPU_CORES} núcleos) — InsightFace buffalo_sc @ 256×192")

LOG_COOLDOWN     = 5
COSINE_THRESHOLD = 0.40   # similitud coseno ≥ → "conocido"

# ── Utilidades ────────────────────────────────────────────────────────────────

def _normalize(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-8)

def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


class RecognitionEngine:
    """
    Motor de reconocimiento facial — Deep Learning end-to-end:
      Detección:      RetinaFace  (buffalo_l/sc — maneja todos los ángulos)
      Identificación: ArcFace     (embeddings coseno, >98% accuracy)
    IMPORTANTE: retrain() usa _face_app.get() igual que process_frame()
    para garantizar que los embeddings sean siempre del mismo espacio.
    """

    def __init__(self):
        from insightface.app import FaceAnalysis

        providers = (
            ['CUDAExecutionProvider', 'CPUExecutionProvider']
            if _CUDA_AVAILABLE else ['CPUExecutionProvider']
        )

        self._face_app = FaceAnalysis(
            name=_MODEL_NAME,
            providers=providers,
            allowed_modules=['detection', 'recognition'],
        )
        self._face_app.prepare(
            ctx_id=0 if _CUDA_AVAILABLE else -1,
            det_size=_DET_SIZE,
        )

        self._known:  dict[int, dict] = {}
        self._lock    = threading.Lock()
        self._trained = False

        # Lock de inferencia — evita que retrain y process_frame usen ONNX al mismo tiempo
        self._infer_lock = threading.Lock()

        self.frame_count   = 0
        self._last_results = []
        self._last_log: dict[str, float] = {}
        self.alert_manager = AlertManager()

        self.retrain()

    # ── Data augmentation ─────────────────────────────────────────────────────

    @staticmethod
    def _augment(img: np.ndarray) -> list:
        """4 variantes: original + espejo + brillo+/- (requerimiento funcional)."""
        return [
            img,
            cv2.flip(img, 1),
            cv2.convertScaleAbs(img, alpha=1.25, beta=35),
            cv2.convertScaleAbs(img, alpha=0.75, beta=-25),
        ]

    # ── Entrenamiento ─────────────────────────────────────────────────────────

    def retrain(self):
        """
        Carga muestras de la DB y construye embeddings promediados por identidad.
        Usa _face_app.get() — MISMO pipeline que process_frame() — para garantizar
        que retrain e inferencia estén en el mismo espacio de embedding.
        Ignora automáticamente muestras LBPH antiguas (100×100 grises).
        """
        samples = get_all_face_samples()
        if not samples:
            with self._lock:
                self._trained = False
                self._known   = {}
            return

        grouped: dict[int, dict] = {}

        for identity_id, name, blob in samples:
            nparr = np.frombuffer(blob, np.uint8)
            img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                continue

            h, w = img.shape[:2]

            # Descartar muestras antiguas LBPH (100×100 grises)
            if (h, w) == (100, 100):
                continue

            # Normalizar tamaño: asegurarse de que hay suficiente contexto para RetinaFace
            # Muestras nuevas: 224×224. Muestras previas (112×112): agregar padding.
            if (h, w) == (112, 112):
                img = cv2.copyMakeBorder(img, 56, 56, 56, 56, cv2.BORDER_REFLECT_101)
            elif h < 112 or w < 112:
                continue   # demasiado pequeño
            # 224×224 o mayor: usar directo

            # Augmentar: 4× GPU, 1× CPU (retrain más rápido en CPU)
            variants = self._augment(img) if _CUDA_AVAILABLE else [img]

            for variant in variants:
                with self._infer_lock:
                    faces = self._face_app.get(variant)
                if not faces:
                    continue
                # Usar face.embedding — idéntico al usado en process_frame
                emb = _normalize(faces[0].embedding)
                if identity_id not in grouped:
                    grouped[identity_id] = {'name': name, 'embs': []}
                grouped[identity_id]['embs'].append(emb)

        if not grouped:
            print("[OmniFace] retrain: sin muestras ArcFace — registra los usuarios de nuevo")
            with self._lock:
                self._trained = False
                self._known   = {}
            return

        known = {}
        for iid, data in grouped.items():
            embs     = np.array(data['embs'])
            mean_emb = _normalize(embs.mean(axis=0))
            known[iid] = {'name': data['name'], 'embedding': mean_emb}
            print(f"[OmniFace] Entrenado: {data['name']} — {len(embs)} embeddings")

        with self._lock:
            self._known   = known
            self._trained = True

        print(f"[OmniFace] retrain completado: {len(known)} identidades")

    # ── Pipeline por frame ────────────────────────────────────────────────────

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        self.frame_count += 1

        if self.frame_count % PROCESS_EVERY_N == 0:
            try:
                with self._infer_lock:
                    faces = self._face_app.get(frame)

                results = []
                for face in faces:
                    emb  = _normalize(face.embedding)
                    name, conf, iid = self._identify(emb)
                    bbox = face.bbox.astype(int)
                    self._try_log(name, iid, conf, frame)
                    results.append({
                        'name':        name,
                        'confidence':  conf,
                        'identity_id': iid,
                        'box':         (bbox[0], bbox[1], bbox[2], bbox[3]),
                        'known':       name != 'Desconocido',
                    })

                self._last_results = results
                self.alert_manager.update(results)

            except Exception as e:
                print(f"[OmniFace] Error en process_frame: {e}")

        return self._draw(frame.copy(), self._last_results)

    # ── Detección para registro ───────────────────────────────────────────────

    def detect_faces(self, frame: np.ndarray) -> list:
        """Detección sin bloquear el hilo principal del registro."""
        try:
            with self._infer_lock:
                return self._face_app.get(frame)
        except Exception:
            return []

    # ── Identificación ────────────────────────────────────────────────────────

    def _identify(self, embedding: np.ndarray):
        with self._lock:
            trained = self._trained
            known   = dict(self._known)

        if not trained or not known:
            return 'Desconocido', 0.0, None

        best_id, best_sim = None, -1.0
        for iid, data in known.items():
            sim = _cosine_sim(embedding, data['embedding'])
            if sim > best_sim:
                best_sim, best_id = sim, iid

        if best_sim >= COSINE_THRESHOLD:
            return known[best_id]['name'], round(best_sim * 100, 1), best_id

        return 'Desconocido', round(max(0.0, best_sim) * 100, 1), None

    # ── Log ───────────────────────────────────────────────────────────────────

    def _try_log(self, name, iid, conf, frame):
        now = time.time()
        if now - self._last_log.get(name, 0) < LOG_COOLDOWN:
            return
        self._last_log[name] = now
        screenshot = self.alert_manager.save_screenshot(frame) if name == 'Desconocido' else None
        log_access(name, 'known' if name != 'Desconocido' else 'unknown', conf, iid, screenshot)

    # ── Dibujo ────────────────────────────────────────────────────────────────

    def _draw(self, frame: np.ndarray, results: list) -> np.ndarray:
        for r in results:
            x1, y1, x2, y2 = r['box']
            color = (
                (50, 220, 100) if r['known'] and r['confidence'] >= 75 else
                (50, 165, 255) if r['known'] else
                (50,  50, 220)
            )
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label      = f"{r['name']}  {r['confidence']}%"
            (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(frame, (x1, y2), (x1 + tw + 10, y2 + 24), color, -1)
            cv2.putText(frame, label, (x1 + 5, y2 + 17),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        return frame

    def shutdown(self):
        pass
