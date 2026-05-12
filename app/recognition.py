import cv2
import numpy as np
import time
import threading

from .database import get_all_face_samples, log_access
from .alerts import AlertManager

# ── Parámetros ────────────────────────────────────────────────────────────────
FACE_SIZE       = (150, 150)  # ↑ de 100 → mayor detalle para LBPH
MAX_DIST        = 70          # distancia LBPH — umbral para "conocido"
PROCESS_EVERY_N = 2           # procesar 1 de cada 2 frames (era 3)
LOG_COOLDOWN    = 5

# ── Cascades ──────────────────────────────────────────────────────────────────
_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
_profile_path = cv2.data.haarcascades + 'haarcascade_profileface.xml'
_eye_path     = cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml'


class RecognitionEngine:
    def __init__(self):
        self.detector         = cv2.CascadeClassifier(_cascade_path)
        self.profile_detector = cv2.CascadeClassifier(_profile_path)
        self.eye_detector     = cv2.CascadeClassifier(_eye_path)

        self.recognizer = cv2.face.LBPHFaceRecognizer_create(
            radius=2, neighbors=8, grid_x=8, grid_y=8
        )
        # radius=2 captura patrones más amplios → mejor para caras parciales

        self._id_map:  dict[int, str] = {}
        self._trained  = False
        self._lock     = threading.Lock()
        self.frame_count   = 0
        self._last_results = []
        self._last_log: dict[str, float] = {}
        self.alert_manager = AlertManager()

        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        self.retrain()

    # ── Preprocesado ──────────────────────────────────────────────────────────

    def _preprocess(self, gray_img):
        """CLAHE para normalizar iluminación."""
        return self.clahe.apply(gray_img)

    def align_face(self, gray_roi):
        """
        Alinea la cara detectando los ojos con Haar cascade y rotando
        para que queden horizontales. Sin dlib — solo OpenCV.
        Retorna la imagen alineada, o la original si no detecta ojos.
        """
        h, w = gray_roi.shape
        min_eye = max(w // 7, 15)
        eyes = self.eye_detector.detectMultiScale(
            gray_roi, scaleFactor=1.1, minNeighbors=4,
            minSize=(min_eye, min_eye)
        )

        if len(eyes) < 2:
            return gray_roi  # sin alineación — mejor que distorsionar

        # Ordenar de izquierda a derecha
        eyes = sorted(eyes, key=lambda e: e[0])
        # Centros de los dos ojos más plausibles (izq y der)
        lx = eyes[0][0] + eyes[0][2] // 2
        ly = eyes[0][1] + eyes[0][3] // 2
        rx = eyes[1][0] + eyes[1][2] // 2
        ry = eyes[1][1] + eyes[1][3] // 2

        # Ángulo de inclinación
        angle = np.degrees(np.arctan2(float(ry - ly), float(rx - lx)))

        if abs(angle) < 1.0:
            return gray_roi  # ya alineada, no rotar innecesariamente

        # Rotar alrededor del punto medio entre ojos
        cx, cy = float((lx + rx) // 2), float((ly + ry) // 2)
        M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        aligned = cv2.warpAffine(gray_roi, M, (w, h),
                                 flags=cv2.INTER_CUBIC,
                                 borderMode=cv2.BORDER_REPLICATE)
        return aligned

    def _prepare_face(self, gray_roi):
        """Pipeline: CLAHE → resize.
        align_face() desactivado: las muestras en DB son 100×100 (muy chicas
        para detectar ojos) → alinear en inferencia pero no en entrenamiento
        genera mismatch de features y nadie es reconocido."""
        enhanced = self._preprocess(gray_roi)
        return cv2.resize(enhanced, FACE_SIZE)

    def _augment(self, img):
        """8 variantes: original + rotaciones ±15°/±25° + bright + dark + flip.
        Las rotaciones son la clave para que el modelo tolere inclinaciones de cabeza."""
        h, w = img.shape[:2]

        def rot(angle):
            M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
            return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

        return [
            img,
            rot(-15), rot(15),
            rot(-25), rot(25),
            cv2.convertScaleAbs(img, alpha=1.25, beta=35),   # más brillante
            cv2.convertScaleAbs(img, alpha=0.75, beta=-25),  # más oscuro
            cv2.flip(img, 1),                                 # espejo
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

            # Preparar: alinear + CLAHE + resize
            prepared = self._prepare_face(img)

            # Aumentar: 4 variantes × muestra = 4× el dataset
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

    # ── Detección de TODAS las caras ──────────────────────────────────────────

    def _detect_all_faces(self, gray):
        """
        Retorna lista de (pose_str, (x,y,w,h)) para TODAS las caras del frame.
        Frontal primero; si no hay ninguna, busca perfiles.
        """
        results = []

        frontal = self.detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50)
        )
        for box in frontal:
            results.append(("Frente", tuple(box)))

        if not results:
            # Perfil derecho
            profiles_r = self.profile_detector.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50)
            )
            for box in profiles_r:
                results.append(("Perfil D", tuple(box)))

            # Perfil izquierdo (imagen volteada)
            flipped = cv2.flip(gray, 1)
            profiles_l = self.profile_detector.detectMultiScale(
                flipped, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50)
            )
            for (x, y, w, h) in profiles_l:
                orig_x = gray.shape[1] - x - w
                results.append(("Perfil I", (orig_x, y, w, h)))

        return results

    # ── Pipeline por frame ────────────────────────────────────────────────────

    def process_frame(self, frame):
        self.frame_count += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.frame_count % PROCESS_EVERY_N == 0:
            all_faces = self._detect_all_faces(gray)
            results   = []

            for pose_name, (x, y, w, h) in all_faces:
                face_roi = gray[y:y+h, x:x+w]
                prepared = self._prepare_face(face_roi)
                name, confidence, identity_id = self._identify(prepared)

                results.append({
                    'name':        name,
                    'confidence':  confidence,
                    'identity_id': identity_id,
                    'box':         (x, y, x + w, y + h),
                    'known':       name != 'Desconocido',
                    'pose':        pose_name,
                })
                self._try_log(name, identity_id, confidence, frame)

            self._last_results = results
            self.alert_manager.update(results)

        return self._draw(frame.copy(), self._last_results)

    # ── Identificación ────────────────────────────────────────────────────────

    def _identify(self, prepared_face):
        with self._lock:
            trained = self._trained
            id_map  = dict(self._id_map)

        if not trained:
            return 'Desconocido', 0.0, None

        try:
            label, raw_conf = self.recognizer.predict(prepared_face)
        except Exception:
            return 'Desconocido', 0.0, None

        if raw_conf <= MAX_DIST:
            # Curva de potencia: dist=0→100%, dist=5→97%, dist=20→91%, dist=MAX_DIST→0%
            # Más realista que lineal y sube el número mostrado para buenos matches
            norm = raw_conf / MAX_DIST          # 0.0 – 1.0
            confidence = round((1.0 - norm) ** 0.3 * 100, 1)
            name = id_map.get(label, 'Desconocido')
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
                color = (50, 220, 100)   # verde — reconocido con alta confianza
            elif r['known']:
                color = (50, 165, 255)   # naranja — reconocido con baja confianza
            else:
                color = (50, 50, 220)    # rojo — desconocido

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            label       = f"{r['name']}  {r['confidence']}%"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(frame, (x1, y2), (x1 + tw + 10, y2 + 24), color, -1)
            cv2.putText(frame, label, (x1 + 5, y2 + 17),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

            cv2.putText(frame, r.get('pose', ''), (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)

        return frame
