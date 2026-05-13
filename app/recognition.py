import cv2
import numpy as np
import time
import threading
import os

from .database import get_all_face_samples, log_access
from .alerts import AlertManager

# ── Parámetros ────────────────────────────────────────────────────────────────
COSINE_THRESHOLD = 0.40   # similitud coseno ArcFace para "conocido" (rango típico 0.3-0.5)
PROCESS_EVERY_N  = 1      # procesar cada frame (InsightFace en GPU lo aguanta)
LOG_COOLDOWN     = 5      # segundos entre logs del mismo nombre

# ── Detección GPU via onnxruntime ─────────────────────────────────────────────
_CUDA_AVAILABLE = False
_GPU_NAME       = "CPU"
try:
    import onnxruntime as _ort
    _providers = _ort.get_available_providers()
    if "CUDAExecutionProvider" in _providers:
        _CUDA_AVAILABLE = True
        try:
            import subprocess
            _GPU_NAME = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                timeout=3
            ).decode().strip().splitlines()[0]
        except Exception:
            _GPU_NAME = "NVIDIA GPU"
except ImportError:
    pass

_CPU_CORES = os.cpu_count() or 2

if _CUDA_AVAILABLE:
    print(f"[OmniFace] GPU detectada: {_GPU_NAME} — InsightFace via CUDAExecutionProvider")
else:
    print(f"[OmniFace] Sin GPU CUDA — InsightFace en CPU ({_CPU_CORES} núcleos)")


# ── Utilidades de embedding ───────────────────────────────────────────────────

def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / (n + 1e-8)


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Similitud coseno entre dos vectores ya normalizados."""
    return float(np.dot(a, b))


class RecognitionEngine:
    """
    Motor de reconocimiento facial basado en InsightFace:
      - Detección:     RetinaFace  (maneja frente + perfil nativamente)
      - Identificación: ArcFace    (embeddings 512-dim, similitud coseno)
      - GPU:           onnxruntime CUDAExecutionProvider si disponible
    """

    def __init__(self):
        from insightface.app import FaceAnalysis

        providers = (
            ['CUDAExecutionProvider', 'CPUExecutionProvider']
            if _CUDA_AVAILABLE else
            ['CPUExecutionProvider']
        )

        self._face_app = FaceAnalysis(
            name='buffalo_l',
            providers=providers,
            allowed_modules=['detection', 'recognition'],
        )
        # ctx_id=0 → GPU 0;  ctx_id=-1 → CPU
        self._face_app.prepare(
            ctx_id=0 if _CUDA_AVAILABLE else -1,
            det_size=(640, 480),
        )

        # Referencia directa al modelo ArcFace para retrain sin re-detectar
        self._rec_model = None
        for m in self._face_app.models.values():
            if hasattr(m, 'get_feat'):
                self._rec_model = m
                break

        # Identidades conocidas: {identity_id: {'name': str, 'embedding': np.ndarray}}
        self._known:   dict[int, dict] = {}
        self._lock     = threading.Lock()
        self._trained  = False

        self.frame_count   = 0
        self._last_results = []
        self._last_log: dict[str, float] = {}
        self.alert_manager = AlertManager()

        self.retrain()

    # ── Data augmentation ─────────────────────────────────────────────────────

    @staticmethod
    def _augment(img: np.ndarray) -> list:
        """
        4 variantes por muestra: original + espejo + más brillo + menos brillo.
        Requerimiento: 'generar automáticamente variaciones de sus fotos'.
        """
        return [
            img,
            cv2.flip(img, 1),                                          # espejo
            cv2.convertScaleAbs(img, alpha=1.25, beta=35),             # más brillante
            cv2.convertScaleAbs(img, alpha=0.75, beta=-25),            # más oscuro
        ]

    # ── Entrenamiento ─────────────────────────────────────────────────────────

    def retrain(self):
        """
        Carga muestras de la DB, extrae embeddings ArcFace con augmentación
        y calcula un embedding promedio normalizado por identidad.
        Solo procesa muestras nuevas (color 112×112) — ignora las antiguas LBPH.
        """
        samples = get_all_face_samples()
        if not samples:
            with self._lock:
                self._trained = False
                self._known   = {}
            return

        if self._rec_model is None:
            print("[OmniFace] retrain: modelo ArcFace no disponible")
            return

        grouped: dict[int, dict] = {}   # {id: {'name': str, 'imgs': []}}

        for identity_id, name, blob in samples:
            nparr = np.frombuffer(blob, np.uint8)
            img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                continue

            h, w = img.shape[:2]
            # Muestras antiguas LBPH eran 100×100 grises → ignorar
            if (h, w) != (112, 112):
                continue

            if identity_id not in grouped:
                grouped[identity_id] = {'name': name, 'imgs': []}
            grouped[identity_id]['imgs'].append(img)

        if not grouped:
            print("[OmniFace] retrain: no hay muestras ArcFace — vuelve a registrar las identidades")
            with self._lock:
                self._trained = False
                self._known   = {}
            return

        known = {}
        for identity_id, data in grouped.items():
            # Augmentar todas las imágenes y apilarlas para inferencia en lote
            all_variants = []
            for img in data['imgs']:
                all_variants.extend(self._augment(img))

            # Inferencia en lote → más rápido en GPU
            feats = self._rec_model.get_feat(all_variants)   # (N, 512)
            embs  = np.array([_normalize(f) for f in feats]) # (N, 512)

            mean_emb = _normalize(embs.mean(axis=0))
            known[identity_id] = {
                'name':      data['name'],
                'embedding': mean_emb,
            }
            print(f"[OmniFace] Entrenado: {data['name']} — {len(all_variants)} vectores")

        with self._lock:
            self._known   = known
            self._trained = True

        print(f"[OmniFace] retrain completado: {len(known)} identidades")

    # ── Pipeline por frame ────────────────────────────────────────────────────

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        self.frame_count += 1

        if self.frame_count % PROCESS_EVERY_N == 0:
            try:
                faces   = self._face_app.get(frame)
                results = []
                for face in faces:
                    emb  = _normalize(face.embedding)
                    name, confidence, identity_id = self._identify(emb)
                    bbox = face.bbox.astype(int)
                    x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
                    self._try_log(name, identity_id, confidence, frame)
                    results.append({
                        'name':        name,
                        'confidence':  confidence,
                        'identity_id': identity_id,
                        'box':         (x1, y1, x2, y2),
                        'known':       name != 'Desconocido',
                    })

                self._last_results = results
                self.alert_manager.update(results)

            except Exception as e:
                print(f"[OmniFace] Error en process_frame: {e}")

        return self._draw(frame.copy(), self._last_results)

    # ── Identificación ────────────────────────────────────────────────────────

    def _identify(self, embedding: np.ndarray):
        with self._lock:
            trained = self._trained
            known   = dict(self._known)

        if not trained or not known:
            return 'Desconocido', 0.0, None

        best_id  = None
        best_sim = -1.0
        for identity_id, data in known.items():
            sim = _cosine_sim(embedding, data['embedding'])
            if sim > best_sim:
                best_sim = sim
                best_id  = identity_id

        if best_sim >= COSINE_THRESHOLD:
            confidence = round(best_sim * 100, 1)
            return known[best_id]['name'], confidence, best_id

        return 'Desconocido', round(max(0.0, best_sim) * 100, 1), None

    # ── Log ───────────────────────────────────────────────────────────────────

    def _try_log(self, name, identity_id, confidence, frame):
        now  = time.time()
        last = self._last_log.get(name, 0)
        if now - last < LOG_COOLDOWN:
            return
        self._last_log[name] = now

        screenshot_path = None
        if name == 'Desconocido':
            screenshot_path = self.alert_manager.save_screenshot(frame)

        log_access(name, 'known' if name != 'Desconocido' else 'unknown',
                   confidence, identity_id, screenshot_path)

    # ── Dibujo ────────────────────────────────────────────────────────────────

    def _draw(self, frame: np.ndarray, results: list) -> np.ndarray:
        for r in results:
            x1, y1, x2, y2 = r['box']

            if r['known'] and r['confidence'] >= 75:
                color = (50, 220, 100)    # verde — reconocido con alta confianza
            elif r['known']:
                color = (50, 165, 255)    # naranja — reconocido con baja confianza
            else:
                color = (50, 50, 220)     # rojo — desconocido

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            label       = f"{r['name']}  {r['confidence']}%"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(frame, (x1, y2), (x1 + tw + 10, y2 + 24), color, -1)
            cv2.putText(frame, label, (x1 + 5, y2 + 17),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        return frame

    # ── Limpieza ──────────────────────────────────────────────────────────────

    def shutdown(self):
        pass  # InsightFace / onnxruntime no requieren cleanup explícito
