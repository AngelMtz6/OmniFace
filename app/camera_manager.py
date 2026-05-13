import cv2
import threading
import time


class CameraWorker:
    """Captura frames de una fuente (índice int o URL RTSP/HTTP) en hilo de fondo."""

    def __init__(self, source, name: str = "Cámara"):
        self.source    = source
        self.name      = name
        self._frame    = None
        self._frame_id = 0
        self._lock     = threading.Lock()
        self._running  = True
        self.connected = False

        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _open(self):
        if isinstance(self.source, int):
            cap = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        else:
            cap = cv2.VideoCapture(str(self.source))
        return cap

    def _loop(self):
        cap = self._open()
        while self._running:
            if not cap.isOpened():
                self.connected = False
                time.sleep(3.0)
                cap = self._open()
                continue

            ret, frame = cap.read()
            if not ret or frame is None:
                self.connected = False
                time.sleep(0.05)
                continue

            # Espejo solo para cámaras locales
            if isinstance(self.source, int):
                frame = cv2.flip(frame, 1)

            self.connected = True
            with self._lock:
                self._frame    = frame
                self._frame_id += 1

    def get_frame(self):
        with self._lock:
            if self._frame is None:
                return None, -1
            return self._frame.copy(), self._frame_id

    def release(self):
        self._running = False


class CameraManager:
    """Gestiona múltiples CameraWorkers (locales + RTSP)."""

    def __init__(self):
        self._workers: dict[int, CameraWorker] = {}
        self._lock = threading.Lock()

    def load_from_db(self):
        from .database import get_cameras
        for cam in get_cameras():
            if cam["active"]:
                self.add(cam["id"], cam["source"], cam["name"])

    def add(self, cam_id: int, source, name: str = "Cámara"):
        """Añade o reemplaza una cámara. Convierte source numérico automáticamente."""
        with self._lock:
            if cam_id in self._workers:
                self._workers[cam_id].release()
            try:
                src = int(source)
            except (ValueError, TypeError):
                src = str(source)
            self._workers[cam_id] = CameraWorker(src, name)

    def remove(self, cam_id: int):
        with self._lock:
            if cam_id in self._workers:
                self._workers[cam_id].release()
                del self._workers[cam_id]

    def ids(self) -> list[int]:
        with self._lock:
            return list(self._workers.keys())

    def get_frames(self) -> dict[int, tuple]:
        with self._lock:
            workers = dict(self._workers)
        return {cid: w.get_frame() for cid, w in workers.items()}

    def get_first_frame(self):
        with self._lock:
            if not self._workers:
                return None, -1
            return next(iter(self._workers.values())).get_frame()

    def get_names(self) -> dict[int, str]:
        with self._lock:
            return {cid: w.name for cid, w in self._workers.items()}

    def get_connected(self) -> dict[int, bool]:
        with self._lock:
            return {cid: w.connected for cid, w in self._workers.items()}

    def release_all(self):
        with self._lock:
            for w in self._workers.values():
                w.release()
            self._workers.clear()
