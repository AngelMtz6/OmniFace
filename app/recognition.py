import cv2
import numpy as np
import time
import threading
import os
import concurrent.futures

from .database import get_all_face_samples, log_access
from .alerts import AlertManager

# ── Parámetros ────────────────────────────────────────────────────────────────
FACE_SIZE       = (150, 150)
MAX_DIST        = 70          # distancia LBPH umbral para "conocido"
PROCESS_EVERY_N = 2
LOG_COOLDOWN    = 5

# ── Cascades ──────────────────────────────────────────────────────────────────
_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
_profile_path = cv2.data.haarcascades + 'haarcascade_profileface.xml'
_eye_path     = cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml'

# ── Detección de GPU ──────────────────────────────────────────────────────────
# opencv-contrib-python NO incluye CUDA en PyPI.
# Usamos onnxruntime-gpu para detectar y usar la GPU correctamente.
_CUDA_AVAILABLE = False
_GPU_NAME       = "CPU"
try:
    import onnxruntime as _ort
    _providers = _ort.get_available_providers()
    if "CUDAExecutionProvider" in _providers:
        _CUDA_AVAILABLE = True
        # Obtener nombre de GPU vía onnxruntime session options si es posible
        try:
            import subprocess, re
            _smi = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                timeout=3
            ).decode().strip().splitlines()[0]
            _GPU_NAME = _smi
        except Exception:
            _GPU_NAME = "NVIDIA GPU"
except ImportError:
    pass

_CPU_CORES = os.cpu_count() or 2

if _CUDA_AVAILABLE:
    print(f"[OmniFace] GPU detectada: {_GPU_NAME} — usando CUDAExecutionProvider (onnxruntime)")
else:
    print(f"[OmniFace] Sin GPU CUDA — CPU paralelo ({_CPU_CORES} nucleos)")


class RecognitionEngine:
    def __init__(self):
        self.detector         = cv2.CascadeClassifier(_cascade_path)
        self.profile_detector = cv2.CascadeClassifier(_profile_path)
        self.eye_detector     = cv2.CascadeClassifier(_eye_path)

        self.recognizer = cv2.face.LBPHFaceRecognizer_create(
            radius=2, neighbors=8, grid_x=8, grid_y=8
        )

        self._id_map:  dict[int, str] = {}
        self._trained  = False
        self._lock     = threading.Lock()
        self._cascade_lock = threading.Lock()   # CascadeClassifier NO es thread-safe
        self.frame_count   = 0
        self._last_results = []
        self._last_log: dict[str, float] = {}
        self.alert_manager = AlertManager()

        # ── Preprocesado CLAHE — siempre CPU (OpenCV sin CUDA en PyPI) ──────────
        # opencv-contrib-python de PyPI NO tiene soporte CUDA compilado.
        # La GPU se usa ÚNICAMENTE vía onnxruntime (InsightFace / modelos ONNX).
        self.clahe     = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self._use_cuda = False   # cv2.cuda no disponible en wheels de PyPI

        # ── Pool de hilos: detección + reconocimiento en paralelo ─────────────
        # max_workers = min(6, cores) para no saturar en laptops
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=min(6, _CPU_CORES)
        )

        self.retrain()

    # ── Preprocesado ──────────────────────────────────────────────────────────

    def _preprocess(self, gray_img):
        """CLAHE en GPU si está disponible, CPU si no."""
        if self._use_cuda:
            gpu = cv2.cuda_GpuMat()
            gpu.upload(gray_img)
            gpu_out = self._cuda_clahe.apply(gpu, self._cuda_stream)
            self._cuda_stream.waitForCompletion()
            return gpu_out.download()
        return self.clahe.apply(gray_img)

    def align_face(self, gray_roi):
        """
        Alinea la cara detectando los ojos con Haar cascade.
        Protegido con lock porque CascadeClassifier no es thread-safe.
        Solo actua si el ROI tiene >=80px — muy chico no detecta ojos.
        """
        h, w = gray_roi.shape
        if min(h, w) < 80:
            return gray_roi

        min_eye = max(w // 7, 15)
        with self._cascade_lock:   # ← evita crash con múltiples hilos
            eyes = self.eye_detector.detectMultiScale(
                gray_roi, scaleFactor=1.1, minNeighbors=4,
                minSize=(min_eye, min_eye)
            )

        if len(eyes) < 2:
            return gray_roi

        eyes = sorted(eyes, key=lambda e: e[0])
        lx = eyes[0][0] + eyes[0][2] // 2;  ly = eyes[0][1] + eyes[0][3] // 2
        rx = eyes[1][0] + eyes[1][2] // 2;  ry = eyes[1][1] + eyes[1][3] // 2

        angle = np.degrees(np.arctan2(float(ry - ly), float(rx - lx)))
        if abs(angle) < 1.0:
            return gray_roi

        cx, cy = float((lx + rx) // 2), float((ly + ry) // 2)
        M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        return cv2.warpAffine(gray_roi, M, (w, h),
                              flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE)

    def _prepare_face(self, gray_roi):
        """CLAHE → resize. Alignment solo si el ROI es suficientemente grande."""
        roi = self.align_face(gray_roi)
        enhanced = self._preprocess(roi)
        return cv2.resize(enhanced, FACE_SIZE)

    def _augment(self, img):
        """8 variantes: original + rotaciones +-15/+-25 + bright + dark + flip.
        Las rotaciones son la clave para tolerar inclinaciones de cabeza en vivo."""
        h, w = img.shape[:2]

        def rot(angle):
            M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
            return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

        return [
            img,
            rot(-15), rot(15),
            rot(-25), rot(25),
            cv2.convertScaleAbs(img, alpha=1.25, beta=35),
            cv2.convertScaleAbs(img, alpha=0.75, beta=-25),
            cv2.flip(img, 1),
        ]

    # ── Entrenamiento ─────────────────────────────────────────────────────────

    def retrain(self):
        samples = get_all_face_samples()
        if not samples:
            with self._lock:
                self._trained = False
                self._id_map  = {}
            return

        faces, labels, id_map = [], [], {}
        for identity_id, name, blob in samples:
            nparr = np.frombuffer(blob, np.uint8)
            img   = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            prepared = self._prepare_face(img)
            for variant in self._augment(prepared):
                faces.append(variant)
                labels.append(identity_id)
            id_map[identity_id] = name

        if not faces:
            return

        recognizer = cv2.face.LBPHFaceRecognizer_create(
            radius=2, neighbors=8, grid_x=8, grid_y=8
        )
        recognizer.train(faces, np.array(labels, dtype=np.int32))

        with self._lock:
            self.recognizer = recognizer
            self._id_map    = id_map
            self._trained   = True

    # ── NMS ───────────────────────────────────────────────────────────────────

    @staticmethod
    def _iou(a, b):
        ax2, ay2 = a[0] + a[2], a[1] + a[3]
        bx2, by2 = b[0] + b[2], b[1] + b[3]
        ix = max(0, min(ax2, bx2) - max(a[0], b[0]))
        iy = max(0, min(ay2, by2) - max(a[1], b[1]))
        inter = ix * iy
        union = a[2]*a[3] + b[2]*b[3] - inter
        return inter / union if union > 0 else 0.0

    def _nms(self, detections, iou_thr=0.35):
        if len(detections) <= 1:
            return detections
        detections = sorted(detections, key=lambda d: d[1][2] * d[1][3], reverse=True)
        kept = []
        for det in detections:
            if not any(self._iou(det[1], k[1]) > iou_thr for k in kept):
                kept.append(det)
        return kept

    # ── Detección paralela de TODAS las caras ─────────────────────────────────

    def _detect_all_faces(self, gray):
        """
        Corre los 3 cascades en paralelo (ThreadPoolExecutor) y deduplica con NMS.
        Detecta caras a 0°, 45° y 90° simultaneamente sin bloquear el hilo principal.
        """
        flipped = cv2.flip(gray, 1)
        w_frame = gray.shape[1]

        def _frontal():
            with self._cascade_lock:
                boxes = self.detector.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
            return [("Frente", tuple(map(int, b))) for b in boxes]

        def _profile_r():
            with self._cascade_lock:
                boxes = self.profile_detector.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
            return [("Perfil D", tuple(map(int, b))) for b in boxes]

        def _profile_l():
            with self._cascade_lock:
                boxes = self.profile_detector.detectMultiScale(
                    flipped, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
            return [("Perfil I", (w_frame - int(x) - int(w), int(y), int(w), int(h)))
                    for (x, y, w, h) in boxes]

        fts = [self._executor.submit(f) for f in (_frontal, _profile_r, _profile_l)]
        detections = []
        for ft in fts:
            detections.extend(ft.result())

        return self._nms(detections)

    # ── Pipeline por frame ────────────────────────────────────────────────────

    def _process_single_face(self, gray, pose_name, box, frame):
        """Procesa una sola cara: preparar → identificar → loguear."""
        x, y, w, h = box
        prepared = self._prepare_face(gray[y:y+h, x:x+w])
        name, confidence, identity_id = self._identify(prepared)
        self._try_log(name, identity_id, confidence, frame)
        return {
            'name':        name,
            'confidence':  confidence,
            'identity_id': identity_id,
            'box':         (x, y, x + w, y + h),
            'known':       name != 'Desconocido',
            'pose':        pose_name,
        }

    def process_frame(self, frame):
        self.frame_count += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.frame_count % PROCESS_EVERY_N == 0:
            try:
                all_faces = self._detect_all_faces(gray)

                if all_faces:
                    fts = [
                        self._executor.submit(self._process_single_face, gray, pose, box, frame)
                        for pose, box in all_faces
                    ]
                    results = []
                    for ft in fts:
                        try:
                            results.append(ft.result(timeout=0.5))
                        except Exception:
                            pass
                else:
                    results = []

                self._last_results = results
                self.alert_manager.update(results)
            except Exception:
                pass  # nunca romper el loop de video

        return self._draw(frame.copy(), self._last_results)

    # ── Identificación ────────────────────────────────────────────────────────

    def _identify(self, prepared_face):
        with self._lock:
            trained    = self._trained
            id_map     = dict(self._id_map)
            recognizer = self.recognizer   # snapshot — seguro si retrain reemplaza el objeto

        if not trained:
            return 'Desconocido', 0.0, None

        try:
            label, raw_conf = recognizer.predict(prepared_face)
        except Exception:
            return 'Desconocido', 0.0, None

        if raw_conf <= MAX_DIST:
            # Curva de potencia: dist=0->100%, dist=5->97%, dist=20->91%, dist=MAX_DIST->0%
            norm       = raw_conf / MAX_DIST
            confidence = round((1.0 - norm) ** 0.3 * 100, 1)
            name       = id_map.get(label, 'Desconocido')
            return name, confidence, label

        return 'Desconocido', 0.0, None

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

    def _draw(self, frame, results):
        for r in results:
            x1, y1, x2, y2 = r['box']

            if r['known'] and r['confidence'] >= 75:
                color = (50, 220, 100)
            elif r['known']:
                color = (50, 165, 255)
            else:
                color = (50, 50, 220)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            label       = f"{r['name']}  {r['confidence']}%"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(frame, (x1, y2), (x1 + tw + 10, y2 + 24), color, -1)
            cv2.putText(frame, label, (x1 + 5, y2 + 17),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

            cv2.putText(frame, r.get('pose', ''), (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)

        return frame

    # ── Limpieza ──────────────────────────────────────────────────────────────

    def shutdown(self):
        """Liberar el pool de hilos al cerrar la app."""
        self._executor.shutdown(wait=False)
