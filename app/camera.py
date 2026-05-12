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
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self._frame = None
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
