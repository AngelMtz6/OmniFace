import cv2
import threading
import time


class VideoCamera:
    """Singleton thread-safe camera handler."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        
        print("DEBUG: Iniciando VideoCamera...")
        self.cap = cv2.VideoCapture(0)
        
        if not self.cap.isOpened():
            print("DEBUG: Index 0 (Default) falló. Intentando con CAP_DSHOW...")
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            
        if not self.cap.isOpened():
            print("DEBUG: CAP_DSHOW falló. Intentando con index 1...")
            self.cap = cv2.VideoCapture(1)

        if not self.cap.isOpened():
            print("ERROR: No se pudo abrir ninguna cámara.")
            self.cap = cv2.VideoCapture()
        else:
            print("DEBUG: Cámara abierta exitosamente.")
            # Intentar leer un frame para confirmar
            ret, frame = self.cap.read()
            if ret:
                print("DEBUG: Frame capturado con éxito en init.")
                self._frame = frame
            else:
                print("DEBUG: No se pudo leer frame en init.")
            
        self._frame_lock = threading.Lock()
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        self._initialized = True

    def _capture_loop(self):
        while self._running:
            ret, frame = self.cap.read()
            if ret:
                with self._frame_lock:
                    self._frame = frame
            time.sleep(0.03)

    def get_frame(self):
        with self._frame_lock:
            if self._frame is None:
                return None
            return self._frame.copy()

    def release(self):
        self._running = False
        self.cap.release()
