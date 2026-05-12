import cv2
import numpy as np
import time
import threading

from .database import get_all_face_samples, log_access
from .alerts import AlertManager

FACE_SIZE    = (100, 100)   # tamaño de normalización de rostros
LBPH_THRESH  = 85           # confianza máxima para considerar "conocido" (menor = más estricto)
PROCESS_EVERY_N = 3         # procesar 1 de cada N frames
LOG_COOLDOWN    = 5         # segundos entre logs de la misma persona

_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'


class RecognitionEngine:
    def __init__(self):
        self.detector   = cv2.CascadeClassifier(_cascade_path)
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self._id_map: dict[int, str] = {}   # identity_id → name
        self._trained   = False
        self._lock      = threading.Lock()
        self.frame_count   = 0
        self._last_results = []
        self._last_log: dict[str, float] = {}
        self.alert_manager = AlertManager()
        self.retrain()

    # ── Entrenamiento ─────────────────────────────────────────────────────

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
            faces.append(img)
            labels.append(identity_id)
            id_map[identity_id] = name

        if not faces:
            return

        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.train(faces, np.array(labels, dtype=np.int32))

        with self._lock:
            self.recognizer = recognizer
            self._id_map    = id_map
            self._trained   = True

    # ── Pipeline por frame ────────────────────────────────────────────────

    def process_frame(self, frame):
        self.frame_count += 1

        if self.frame_count % PROCESS_EVERY_N == 0:
            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rects = self.detector.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
            )

            results = []
            for (x, y, w, h) in rects:
                name, confidence, identity_id = self._identify(gray[y:y+h, x:x+w])
                results.append({
                    'name':        name,
                    'confidence':  confidence,
                    'identity_id': identity_id,
                    'box':         (x, y, x + w, y + h),
                    'known':       name != 'Desconocido',
                })
                self._try_log(name, identity_id, confidence, frame)

            self._last_results = results
            self.alert_manager.update(results)

        return self._draw(frame.copy(), self._last_results)

    # ── Identificación ────────────────────────────────────────────────────

    def _identify(self, face_roi):
        with self._lock:
            trained  = self._trained
            id_map   = dict(self._id_map)

        if not trained:
            return 'Desconocido', 0.0, None

        face = cv2.resize(face_roi, FACE_SIZE)
        try:
            label, raw_conf = self.recognizer.predict(face)
        except Exception:
            return 'Desconocido', 0.0, None

        if raw_conf <= LBPH_THRESH:
            # Convertir confianza LBPH (menor=mejor) a porcentaje (mayor=mejor)
            confidence = round(max(0.0, 100.0 - raw_conf), 1)
            name = id_map.get(label, 'Desconocido')
            return name, confidence, label

        return 'Desconocido', round(max(0.0, 100.0 - raw_conf), 1), None

    def _try_log(self, name, identity_id, confidence, frame):
        now  = time.time()
        last = self._last_log.get(name, 0)
        if now - last < LOG_COOLDOWN:
            return
        self._last_log[name] = now

        screenshot_path = None
        if name == 'Desconocido':
            screenshot_path = self.alert_manager.save_screenshot(frame)

        status = 'known' if name != 'Desconocido' else 'unknown'
        log_access(name, status, confidence, identity_id, screenshot_path)

    # ── Dibujo ────────────────────────────────────────────────────────────

    def _draw(self, frame, results):
        for r in results:
            x1, y1, x2, y2 = r['box']
            color = (50, 205, 100) if r['known'] else (50, 50, 220)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            label        = f"{r['name']}  {r['confidence']}%"
            (tw, th), _  = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(frame, (x1, y2), (x1 + tw + 10, y2 + 24), color, -1)
            cv2.putText(frame, label, (x1 + 5, y2 + 17),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        return frame
