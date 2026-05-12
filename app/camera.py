import cv2
import threading
import time


class VideoCamera:
    """Singleton thread-safe camera handler."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, video_source=1):
        if self._initialized:
            return

        self.video_source = video_source
        self.cap = cv2.VideoCapture(self.video_source, cv2.CAP_DSHOW)

        if not self.cap.isOpened():
            print(f"Error: No se pudo abrir la cámara {self.video_source}. Intentando con 0...")
            self.video_source = 0
            self.cap.open(self.video_source, cv2.CAP_DSHOW)

        # Configuración de resolución
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # Forzar formato MJPG para evitar frames corruptos
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

        self._frame = None
        self._frame_id = 0
        self._frame_lock = threading.Lock()
        self._running = True

        # Iniciar hilo de captura
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

        self._initialized = True

    def _capture_loop(self):
        while self._running:
            if not self.cap.isOpened():
                time.sleep(1.0)
                self.cap.open(self.video_source, cv2.CAP_DSHOW)
                continue

            ret, frame = self.cap.read()

            if not ret or frame is None:
                time.sleep(0.1)
                continue

            # Voltear horizontalmente (efecto espejo)
            frame = cv2.flip(frame, 1)

            with self._frame_lock:
                self._frame = frame
                self._frame_id += 1

            time.sleep(0.01)  # Máximo 100 FPS en el hilo de captura

    def get_frame(self):
        """Retorna el último fotograma y su ID."""
        with self._frame_lock:
            if self._frame is None:
                return None, -1
            return self._frame.copy(), self._frame_id

    def release(self):
        self._running = False
        if self.cap.isOpened():
            self.cap.release()