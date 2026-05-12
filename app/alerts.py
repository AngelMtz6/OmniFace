import os
import cv2
import time
import threading
from datetime import datetime

SCREENSHOT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'screenshots')
UNKNOWN_THRESHOLD = 3
ALERT_RESET_SECONDS = 10


class AlertManager:
    def __init__(self):
        os.makedirs(SCREENSHOT_DIR, exist_ok=True)
        self.consecutive_unknowns = 0
        self.alert_active = False
        self.last_alert_time = 0
        self._lock = threading.Lock()

    def update(self, results):
        with self._lock:
            has_unknown = any(not r['known'] for r in results)
            has_known   = any(r['known']     for r in results)

            if has_unknown and not has_known:
                self.consecutive_unknowns += 1
            else:
                self.consecutive_unknowns = 0

            now = time.time()
            if self.consecutive_unknowns >= UNKNOWN_THRESHOLD:
                if now - self.last_alert_time > ALERT_RESET_SECONDS:
                    self.alert_active = True
                    self.last_alert_time = now
            else:
                self.alert_active = False

    def save_screenshot(self, frame):
        ts = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        path = os.path.join(SCREENSHOT_DIR, f'unknown_{ts}.jpg')
        cv2.imwrite(path, frame)
        return path

    def get_status(self):
        with self._lock:
            return {
                'alert_active':         self.alert_active,
                'consecutive_unknowns': self.consecutive_unknowns,
            }
