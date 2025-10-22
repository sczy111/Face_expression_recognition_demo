# frame_bus.py
import threading
from typing import Optional
import numpy as np

class FrameBus:
    def __init__(self):
        self._lock = threading.Lock()
        self._frame: Optional[np.ndarray] = None  # RGB uint8
        self._running: bool = False               # gate UI refreshes

    def publish(self, frame_rgb: np.ndarray):
        with self._lock:
            self._frame = frame_rgb

    def pull(self) -> Optional[np.ndarray]:
        with self._lock:
            if not self._running:
                return None
            return None if self._frame is None else self._frame.copy()

    def clear(self):
        with self._lock:
            self._frame = None

    def set_running(self, v: bool):
        with self._lock:
            self._running = v

BUS = FrameBus()
