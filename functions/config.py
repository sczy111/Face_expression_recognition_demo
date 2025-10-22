# config.py
from dataclasses import dataclass, field
from threading import Lock

@dataclass
class AppConfig:
    # model / io
    nclasses: int = 8               
    camera_index: int = 0
    output_path: str = ""          

    # overlay toggles
    show_bbox: bool = True
    show_face: bool = False
    show_va: bool = False
    viz_progress: bool = False
    viz_probs: bool = False
    bars_around_bbox: bool = False
    label_only: bool = False
    bars_with_titles_only: bool = False
    bars_labels_only: bool = False
    model_path: str = ""   # full path to selected .pth under ./models; empty => use default pretrained/emonet_{n}.pth
    _capture_requested: bool = False  # control flags (for starting / stopping capture)
    _stop: bool = False
    compare_enabled: bool = False
    compare_model_path: str = ""
    show_emoji: bool = False


    # internals
    _lock: Lock = field(default_factory=Lock, repr=False, compare=False)

    # public api
    def update(self, **kwargs):
        """Thread-safe bulk update of public fields."""
        with self._lock:
            for k, v in kwargs.items():
                if hasattr(self, k):
                    setattr(self, k, v)

    def snapshot(self) -> dict:
        """Thread-safe read of current config (copy of public fields)."""
        with self._lock:
            return {
                "nclasses": self.nclasses,
                "camera_index": self.camera_index,
                "output_path": self.output_path,
                "show_bbox": self.show_bbox,
                "show_face": self.show_face,
                "show_va": self.show_va,
                "viz_progress": self.viz_progress,
                "viz_probs": self.viz_probs,
                "bars_around_bbox": self.bars_around_bbox,
                "label_only": self.label_only,
                "bars_with_titles_only": self.bars_with_titles_only,
                "bars_labels_only": self.bars_labels_only,
                "_stop": self._stop,
                "model_path": self.model_path,
                "compare_enabled":self.compare_enabled,
                "compare_model_path":self.compare_model_path,
                "show_emoji": self.show_emoji,

            }

    # capture control
    def request_start_capture(self):
        """Called by UI: ask the main loop to start capture once."""
        with self._lock:
            self._capture_requested = True
            self._stop = False

    def request_stop_capture(self):
        """Called by UI: tell the running capture loop to stop."""
        with self._lock:
            self._stop = True
            self._capture_requested = False

    def consume_start_request(self) -> bool:
        """
        Called by main.py: returns True once when a start was requested,
        and clears the request so it won't retrigger.
        """
        with self._lock:
            if self._capture_requested:
                self._capture_requested = False
                return True
            return False

    def should_stop(self) -> bool:
        """Called inside the OpenCV loop to check for a stop signal."""
        with self._lock:
            return self._stop

    # legacy helpers (optional)
    def request_stop(self):
        self.request_stop_capture()

    def clear_stop(self):
        with self._lock:
            self._stop = False
