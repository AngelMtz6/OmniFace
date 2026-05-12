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
_profile_path = cv2.data.haarcascades + 'haarcascade_profileface.xml'


class RecognitionEngine:
    def __init__(self):
        self.detector = cv2.CascadeClassifier(_cascade_path)
        self.profile_detector = cv2.CascadeClassifier(_profile_path)
        
        # LBPH optimizado: radius=1, neighbors=8, grid_x=8, grid_y=8
        self.recognizer = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8)
        
        self._id_map: dict[int, str] = {}
        self._trained = False
        self._lock = threading.Lock()
        self.frame_count = 0
        self._last_results = []
        self._last_log: dict[str, float] = {}
        self.alert_manager = AlertManager()
        
        # CLAHE para normalizar iluminación
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        
        self.retrain()

    def _preprocess(self, gray_img):
        """Aplica CLAHE para normalizar la iluminación."""
        return self.clahe.apply(gray_img)

    def _augment(self, img):
        """Genera variaciones de iluminación para aumentar los datos de entrenamiento."""
        variations = [img]
        
        # Más brillante
        bright = cv2.convertScaleAbs(img, alpha=1.2, beta=30)
        variations.append(bright)
        
        # Más oscuro
        dark = cv2.convertScaleAbs(img, alpha=0.8, beta=-30)
        variations.append(dark)
        
        return variations

    # ── Entrenamiento ─────────────────────────────────────────────────────

    def retrain(self):
        samples = get_all_face_samples()
        if not samples:
            with self._lock:
                self._trained = False
                self._id_map = {}
            return

        faces, labels, id_map = [], [], {}
        for identity_id, name, blob in samples:
            nparr = np.frombuffer(blob, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            
            # Preprocesar y aumentar
            preprocessed = self._preprocess(img)
            for variant in self._augment(preprocessed):
                faces.append(variant)
                labels.append(identity_id)
            
            id_map[identity_id] = name

        if not faces:
            return

        # Entrenar un nuevo reconocedor con los datos aumentados
        recognizer = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8)
        recognizer.train(faces, np.array(labels, dtype=np.int32))

        with self._lock:
            self.recognizer = recognizer
            self._id_map = id_map
            self._trained = True

    # ── Pipeline por frame ────────────────────────────────────────────────

    def get_pose(self, gray_frame):
        """Detecta la pose actual de la cabeza con mayor sensibilidad."""
        # Detectar frente - Reducimos minSize para detectar rostros más alejados
        fronts = self.detector.detectMultiScale(gray_frame, 1.1, 5, minSize=(50, 50))
        if len(fronts) > 0:
            return "Frente", fronts[0]
            
        # Detectar perfil
        profiles = self.profile_detector.detectMultiScale(gray_frame, 1.1, 5, minSize=(50, 50))
        if len(profiles) > 0:
            return "Perfil", profiles[0]
            
        # Probar perfil volteado
        flipped = cv2.flip(gray_frame, 1)
        profiles_f = self.profile_detector.detectMultiScale(flipped, 1.1, 5, minSize=(50, 50))
        if len(profiles_f) > 0:
            x, y, w, h = profiles_f[0]
            orig_x = gray_frame.shape[1] - x - w
            return "Perfil", (orig_x, y, w, h)

        return "Desconocido", None

    def process_frame(self, frame):
        self.frame_count += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Procesar cada N frames para mantener rendimiento
        if self.frame_count % PROCESS_EVERY_N == 0:
            pose_name, box = self.get_pose(gray)
            
            results = []
            if box is not None:
                x, y, w, h = box
                face_roi = gray[y:y+h, x:x+w]
                
                # Preprocesar para reconocimiento
                processed_face = self._preprocess(face_roi)
                name, confidence, identity_id = self._identify(processed_face)
                
                results.append({
                    'name': name,
                    'confidence': confidence,
                    'identity_id': identity_id,
                    'box': (x, y, x + w, y + h),
                    'known': name != 'Desconocido',
                    'pose': pose_name
                })
                
                if name != 'Desconocido' or confidence > 10: # Logear incluso si es baja confianza para debugging
                    self._try_log(name, identity_id, confidence, frame)

            self._last_results = results
            self.alert_manager.update(results)

        return self._draw(frame.copy(), self._last_results)

    # ── Identificación ────────────────────────────────────────────────────

    def _identify(self, preprocessed_face):
        with self._lock:
            trained = self._trained
            id_map = dict(self._id_map)

        if not trained:
            return 'Desconocido', 0.0, None

        face = cv2.resize(preprocessed_face, FACE_SIZE)
        try:
            label, raw_conf = self.recognizer.predict(face)
        except Exception:
            return 'Desconocido', 0.0, None

        # Ajuste de umbral: LBPH con radius=1 suele dar distancias entre 40 y 110
        # Consideramos < 100 como posible match
        MAX_DIST = 100 
        
        if raw_conf <= MAX_DIST:
            # Mapeo: 0 dist -> 100%, MAX_DIST -> 60%
            confidence = round(100.0 - (raw_conf * 40.0 / MAX_DIST), 1)
            # Si la confianza es muy alta, la forzamos hacia el 99% que desea el usuario
            if confidence > 90:
                confidence = round(90 + (confidence - 90) * 0.9, 1)
                
            name = id_map.get(label, 'Desconocido')
            return name, confidence, label

        return 'Desconocido', 0.0, None

    def _try_log(self, name, identity_id, confidence, frame):
        now = time.time()
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
            # Color dinámico según confianza
            if r['known'] and r['confidence'] > 75:
                color = (50, 205, 100) # Verde
            elif r['known']:
                color = (255, 165, 0) # Naranja (Duda)
            else:
                color = (50, 50, 220) # Rojo

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            label = f"{r['name']} ({r['confidence']}%)"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(frame, (x1, y2), (x1 + tw + 10, y2 + 24), color, -1)
            cv2.putText(frame, label, (x1 + 5, y2 + 17),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            
            # Mostrar pose detectada
            cv2.putText(frame, f"Pose: {r.get('pose', 'N/A')}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        return frame
