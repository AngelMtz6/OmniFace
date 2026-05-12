import cv2
import numpy as np
import face_recognition
import time
import threading

from .database import get_all_encodings, log_access
from .alerts import AlertManager

TOLERANCE = 0.50          # 0 = estricto, 1 = permisivo
PROCESS_EVERY_N = 3       # procesar 1 de cada N frames (rendimiento)
LOG_COOLDOWN = 5          # segundos entre logs de la misma persona


class RecognitionEngine:
    def __init__(self):
        self.known_encodings = []
        self._enc_lock = threading.Lock()
        self.frame_count = 0
        self._last_results = []
        self._last_log: dict[str, float] = {}
        self.alert_manager = AlertManager()
        self.reload_encodings()

    def reload_encodings(self):
        encodings = get_all_encodings()
        with self._enc_lock:
            self.known_encodings = encodings

    def process_frame(self, frame):
        self.frame_count += 1

        if self.frame_count % PROCESS_EVERY_N == 0:
            # Escalar a mitad para mayor velocidad
            small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            rgb   = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

            locations = face_recognition.face_locations(rgb, model='hog')
            encodings = face_recognition.face_encodings(rgb, locations)

            results = []
            for enc, loc in zip(encodings, locations):
                name, confidence, identity_id = self._identify(enc)
                top, right, bottom, left = [v * 2 for v in loc]
                results.append({
                    'name':        name,
                    'confidence':  confidence,
                    'identity_id': identity_id,
                    'box':         (left, top, right, bottom),
                    'known':       name != 'Desconocido',
                })
                self._try_log(name, identity_id, confidence, frame)

            self._last_results = results
            self.alert_manager.update(results)

        return self._draw(frame.copy(), self._last_results)

    # ------------------------------------------------------------------
    def _identify(self, encoding):
        with self._enc_lock:
            known = list(self.known_encodings)

        if not known:
            return 'Desconocido', 0.0, None

        known_encs = [np.array(k['encoding']) for k in known]
        distances  = face_recognition.face_distance(known_encs, encoding)
        best_idx   = int(np.argmin(distances))
        best_dist  = float(distances[best_idx])

        if best_dist <= TOLERANCE:
            confidence = round((1 - best_dist) * 100, 1)
            match = known[best_idx]
            return match['name'], confidence, match['id']

        return 'Desconocido', round((1 - best_dist) * 100, 1), None

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

    def _draw(self, frame, results):
        for r in results:
            left, top, right, bottom = r['box']
            color = (50, 205, 100) if r['known'] else (50, 50, 220)

            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

            label      = f"{r['name']}  {r['confidence']}%"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(frame, (left, bottom), (left + tw + 10, bottom + 24), color, -1)
            cv2.putText(frame, label, (left + 5, bottom + 17),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        return frame
