# emonet_core.py
from pathlib import Path
from typing import Dict
import numpy as np
import torch
import cv2
import logging
from functions.config import AppConfig
from face_alignment.detection.sfd.sfd_detector import SFDDetector
from emonet.models import EmoNet
from functions.visualizations import (
    draw_progress_bars, draw_prob_barchart,
    draw_bbox_and_label, draw_face_landmarks, draw_valence_arousal_circumplex,
    draw_bars_around_bbox, draw_label_only, draw_bars_with_titles_only, draw_bars_labels_only, draw_emoji_right_of_label,draw_bbox
)
from functions.frame_bus import BUS
from functions.paths import APP_DIR, resource_path


# helpers
def _set_camera_props(cap, w=None, h=None, fps=None, prefer_mjpg=True):
    """Try to apply camera properties and return the (actual_w, actual_h, actual_fps)."""
    # On Windows/virtual cams, MJPG often allows higher resolutions.
    if prefer_mjpg:
        try:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        except Exception:
            pass

    if w is not None: cap.set(cv2.CAP_PROP_FRAME_WIDTH,  int(w))
    if h is not None: cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(h))
    if fps is not None: cap.set(cv2.CAP_PROP_FPS, int(fps))

    # read back what we actually got
    actual_w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    return actual_w, actual_h, actual_fps

def _placeholder_panel(h: int, w: int, label: str = "") -> np.ndarray:
    """Gray panel with an optional label (BGR)."""
    panel = np.full((h, w, 3), 40, dtype=np.uint8)
    if label:
        cv2.putText(panel, label, (10, min(h - 10, 40)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2, cv2.LINE_AA)
    return panel


def _classes_map(n: int):
    if n == 8:
        return {0:"Neutral",1:"Happy",2:"Sad",3:"Surprise",4:"Fear",5:"Disgust",6:"Anger",7:"Contempt"}
    if n == 5:
        return {0:"Neutral",1:"Happy",2:"Sad",3:"Surprise",4:"Anger"}
    return {i: f"Class{i}"}  # fallback

def _weights_path(n: int) -> Path:
    return APP_DIR / "pretrained" / f"emonet_{n}.pth"

def _infer_nclasses_from_filename(p: Path, default_n: int) -> int:
    name = p.stem.lower()
    if "_8" in name or "8" == name[-1]:
        return 8
    if "_5" in name or "5" == name[-1]:
        return 5
    return default_n

def _letterbox_to(img, tw, th):
    h, w = img.shape[:2]
    scale = min(tw / w, th / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (nw, nh)) if (nw != w or nh != h) else img
    h2, w2 = resized.shape[:2]
    pad_r, pad_b = tw - w2, th - h2
    if pad_r < 0 or pad_b < 0:
        return cv2.resize(resized, (tw, th))
    if pad_r or pad_b:
        out = np.zeros((h2 + pad_b, w2 + pad_r, 3), dtype=np.uint8)
        out[:h2, :w2] = resized
        return out
    return resized

class EmoNetApp:
    def __init__(self, cfg: AppConfig, window_name: str = "EmoNet Live"):
        self.cfg = cfg
        self.window_name = window_name

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        logging.getLogger("EmoNetApp").info(
            f"Using device: {self.device} | torch={torch.__version__} | "
            f"CUDA available={torch.cuda.is_available()} | "
            f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU backend'}"
        )
        self.image_size = 256

        self.emonet = None
        self.sfd = None
        self.emotion_classes = {}
        self.current_nclasses = None

        # recording
        self.writer = None
        self.lock_w = None
        self.lock_h = None

        self.emonet_cmp = None
        self.current_nclasses_cmp = None
        self.emotion_classes_cmp = {}
        self._current_weights_path_cmp = None

        torch.backends.cudnn.benchmark = True

    def _load_models_if_needed(self, nclasses: int, model_path: str = ""):
        # decide which weights to load
        path = None
        if model_path:
            path = Path(model_path)
            if not path.exists():
                logging.getLogger("EmoNetApp").warning(f"Model not found: {path}. Falling back to pretrained/emonet_{nclasses}.pth")
                path = None

        if path is None:
            path = _weights_path(nclasses)
        else:
            # allow model file to override nclasses if detectable
            nclasses = _infer_nclasses_from_filename(path, nclasses)

        # if already loaded and same nclasses & same file, skip
        if (self.emonet is not None
            and self.current_nclasses == nclasses
            and getattr(self, "_current_weights_path", None) == str(path)):
            return

        logging.getLogger("EmoNetApp").info(f"Loading EmoNet from {path} (nclasses={nclasses})")

        sd = torch.load(str(path), map_location="cpu")
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
        net = EmoNet(n_expression=nclasses).to(self.device)
        net.load_state_dict(sd, strict=False)
        net.eval()
        self.emonet = net

        self.sfd = SFDDetector(self.device)
        self.emotion_classes = _classes_map(nclasses)
        self.current_nclasses = nclasses
        self._current_weights_path = str(path)

        self._close_writer()

    def _load_compare_if_needed(self, model_path: str):
        """Load/refresh the comparison model if enabled and path changed."""
        if not model_path:
            # no compare model configured
            self.emonet_cmp = None
            self.current_nclasses_cmp = None
            self.emotion_classes_cmp = {}
            self._current_weights_path_cmp = None
            return

        p = Path(model_path)
        if not p.exists():
            logging.getLogger("EmoNetApp").warning(f"Compare model not found: {p}. Comparison disabled.")
            self.emonet_cmp = None
            self.current_nclasses_cmp = None
            self.emotion_classes_cmp = {}
            self._current_weights_path_cmp = None
            return

        # infer 5/8 from filename
        ncmp = _infer_nclasses_from_filename(p, default_n=8)

        # if already loaded and same file, skip
        if (self.emonet_cmp is not None
            and self.current_nclasses_cmp == ncmp
            and self._current_weights_path_cmp == str(p)):
            return

        logging.getLogger("EmoNetApp").info(f"Loading Comparison Model from {p} (nclasses={ncmp})")

        sd = torch.load(str(p), map_location="cpu")
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
        net = EmoNet(n_expression=ncmp).to(self.device)
        net.load_state_dict(sd, strict=False)
        net.eval()

        self.emonet_cmp = net
        self.current_nclasses_cmp = ncmp
        self.emotion_classes_cmp = _classes_map(ncmp)
        self._current_weights_path_cmp = str(p)



    @torch.no_grad()
    def _run_emonet(self, face_rgb: np.ndarray) -> Dict[str, torch.Tensor]:
        img = cv2.resize(face_rgb, (self.image_size, self.image_size))
        ten = torch.from_numpy(img).permute(2,0,1).float().to(self.device) / 255.0
        return self.emonet(ten.unsqueeze(0))

    def _ensure_writer(self, vis: np.ndarray, out_stem: str, fps: float = 60.0) -> np.ndarray:
        if not out_stem:
            return vis
        h, w = vis.shape[:2]
        """if self.lock_w is None or self.lock_h is None:
            self.lock_h, self.lock_w = h, w"""
        fixed = _letterbox_to(vis, self.lock_w, self.lock_h)
        if self.writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out_path = (APP_DIR / out_stem).with_suffix(".mp4")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            self.writer = cv2.VideoWriter(str(out_path), fourcc, fps, (self.lock_w, self.lock_h))
        self.writer.write(fixed)
        return fixed

    def _close_writer(self):
        if self.writer is not None:
            self.writer.release()
        self.writer = None
        self.lock_w = self.lock_h = None

    def run_capture_loop(self):
        log = logging.getLogger("EmoNetApp")

        # load model once before entering loop (will hot-reload if cfg changes)
        snap = self.cfg.snapshot()
        self._load_models_if_needed(snap["nclasses"], model_path=getattr(self.cfg, "model_path", ""))
        if snap.get("compare_enabled", False):
            self._load_compare_if_needed(snap.get("compare_model_path", ""))
        else:
            self.emonet_cmp = None
            self.current_nclasses_cmp = None
            self.emotion_classes_cmp = {}
            self._current_weights_path_cmp = None
        BUS.set_running(True)

        # open camera
        current_cam_index = snap["camera_index"]
        cap = cv2.VideoCapture(current_cam_index)
        desired_w, desired_h, desired_fps = 800, 460, 24
        aw, ah, afps = _set_camera_props(cap, desired_w, desired_h, desired_fps)
        logging.getLogger("EmoNetApp").info(
            f"Capture props requested: {desired_w}x{desired_h}@{desired_fps} → actual: {aw}x{ah}@{afps:.1f}"
        )
        if not cap.isOpened():
            log.error(f"Could not open camera index {snap['camera_index']}")
            BUS.set_running(False)
            return

        log.info(f"✅ Starting capture on camera index {snap['camera_index']}")
        frame_count = 0

        try:
            while True:
                if snap.get("compare_enabled", False):
                    self._load_compare_if_needed(snap.get("compare_model_path", ""))
                else:
                    self.emonet_cmp = None
                    self.current_nclasses_cmp = None
                    self.emotion_classes_cmp = {}
                    self._current_weights_path_cmp = None
                # stop signal?
                if self.cfg.should_stop():
                    log.info("🛑 Stop requested. Exiting capture loop...")
                    break

                snap = self.cfg.snapshot()

                
                if snap["camera_index"] != current_cam_index:
                    log.info(f"Switching camera: {current_cam_index} → {snap['camera_index']}")
                    cap.release()
                    current_cam_index = snap["camera_index"]
                    cap = cv2.VideoCapture(current_cam_index)
                    aw, ah, afps = _set_camera_props(cap, desired_w, desired_h, desired_fps)
                    continue

                ok, frame_bgr = cap.read()
                if not ok:
                    log.warning("Failed to read from camera.")
                    break

                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

                # hot snapshot + (re)load models if needed
                snap = self.cfg.snapshot()
                self._load_models_if_needed(snap["nclasses"], model_path=getattr(self.cfg, "model_path", ""))

                # base left column every frame (always present)
                left_bgr = frame_rgb[:, :, ::-1].astype(np.uint8)
                visualization = left_bgr.copy()

                left_h = left_bgr.shape[0]
                sub_h = left_h // 2               # stacked squares height
                sub_w = sub_h                      # make them square

                # face detection + (optional) left overlays
                with torch.no_grad():
                    faces = self.sfd.detect_from_image(frame_bgr)

                have_pred = False
                pred = None
                val, aro = 0.0, 0.0
                face_crop = None
                have_pred_cmp = False

                if len(faces) > 0:
                    x1, y1, x2, y2, _ = faces[0]
                    x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
                    h, w = frame_rgb.shape[:2]
                    x1 = max(0, min(x1, w - 1)); x2 = max(0, min(x2, w - 1))
                    y1 = max(0, min(y1, h - 1)); y2 = max(0, min(y2, h - 1))

                    if x2 > x1 and y2 > y1:
                        face_crop = frame_rgb[y1:y2, x1:x2, :]
                        pred = self._run_emonet(face_crop.copy())
                        have_pred = True
                        val = float(pred["valence"].clamp(-1.0, 1.0).cpu().item())
                        aro = float(pred["arousal"].clamp(-1.0, 1.0).cpu().item())
                        pred_cmp = None
                        val_cmp, aro_cmp = 0.0, 0.0
                        if snap.get("compare_enabled", False) and self.emonet_cmp is not None:
                            pred_cmp = self._run_emonet(face_crop.copy())
                            # only visualize bars/probs for compare; no left-image overlays
                            val_cmp = float(pred_cmp["valence"].clamp(-1.0, 1.0).cpu().item())
                            aro_cmp = float(pred_cmp["arousal"].clamp(-1.0, 1.0).cpu().item())
                            have_pred_cmp = True
                        # LEFT overlays on the main image (only if we have a prediction)
                        if snap["show_bbox"]:
                            visualization = draw_bbox(
                                visualization, np.array([x1, y1, x2, y2])
                            )
                        if snap["label_only"]:
                            visualization = draw_label_only(
                                visualization, np.array([x1, y1, x2, y2]), pred, self.emotion_classes
                            )
                        # If emoji toggle is on, paste emoji to the right of the label
                        if snap.get("show_emoji", False):
                            visualization = draw_emoji_right_of_label(
                                visualization, np.array([x1, y1, x2, y2]), pred, self.emotion_classes,
                                fixed_offset_x=-85,  # increasing/decreasing to fine-tune
                                fixed_offset_y=-20   # adjust vertical alignment
                            )

                        if snap["bars_around_bbox"]:
                            visualization = draw_bars_around_bbox(
                                visualization, np.array([x1, y1, x2, y2]), val, aro
                            )
                        if snap["bars_with_titles_only"]:
                            visualization = draw_bars_with_titles_only(
                                visualization, np.array([x1, y1, x2, y2]), val, aro
                            )
                        if snap["bars_labels_only"]:
                            visualization = draw_bars_labels_only(
                                visualization, np.array([x1, y1, x2, y2])
                            )

                # Face and VA column (always built when either toggle is on)
                if snap["show_face"] or snap["show_va"]:
                    middle_col = np.zeros((left_h, sub_w, 3), dtype=np.uint8)

                    # TOP: face (or placeholder)
                    if snap["show_face"]:
                        if have_pred and face_crop is not None:
                            face_panel_rgb = draw_face_landmarks(face_crop, pred)           
                            face_panel_bgr = cv2.cvtColor(face_panel_rgb, cv2.COLOR_RGB2BGR) 
                            face_panel_bgr = cv2.resize(face_panel_bgr, (sub_w, sub_h))
                        else:
                            face_panel_bgr = _placeholder_panel(sub_h, sub_w, "face")
                    else:
                        face_panel_bgr = _placeholder_panel(sub_h, sub_w, "")
                    middle_col[:sub_h] = face_panel_bgr

                    # BOTTOM: VA (center dot when no pred) or placeholder if toggle off
                    if snap["show_va"]:
                        v = val if have_pred else 0.0
                        a = aro if have_pred else 0.0
                        va_panel_bgr = draw_valence_arousal_circumplex(v, a, sub_h)
                    else:
                        va_panel_bgr = _placeholder_panel(sub_h, sub_w, "")
                    middle_col[sub_h:sub_h * 2] = va_panel_bgr

                    # attach the middle column once
                    tmp = np.zeros((left_h, visualization.shape[1] + sub_w, 3), dtype=np.uint8)
                    tmp[:, :visualization.shape[1]] = visualization
                    tmp[:, visualization.shape[1]:] = middle_col
                    visualization = tmp

                # VA progress column (zeros when no pred)
                if snap["viz_progress"]:
                    v_main = val if have_pred else 0.0
                    a_main = aro if have_pred else 0.0

                    if snap.get("compare_enabled", False) and self.emonet_cmp is not None:
                        v_cmp = val_cmp if have_pred and have_pred_cmp else 0.0
                        a_cmp = aro_cmp if have_pred and have_pred_cmp else 0.0
                    else:
                        v_cmp = None
                        a_cmp = None

                    bars_panel = draw_progress_bars(
                        v_main, a_main, v_cmp, a_cmp, left_h
                    )

                    tmp = np.zeros((left_h, visualization.shape[1] + bars_panel.shape[1], 3), dtype=np.uint8)
                    tmp[:, :visualization.shape[1]] = visualization
                    tmp[:, visualization.shape[1]:] = bars_panel
                    visualization = tmp


                # class probability column (zeros when no pred)
                if snap["viz_probs"]:
                    names_main = [self.emotion_classes[i] for i in range(self.current_nclasses)]
                    sm_main = (torch.softmax(pred["expression"], dim=1).cpu().numpy()[0]
                            if have_pred else np.zeros(len(names_main), dtype=np.float32))

                    # COMPARE (if enabled)
                    if snap.get("compare_enabled", False) and self.emonet_cmp is not None:
                        names_cmp = [self.emotion_classes_cmp[i] for i in range(self.current_nclasses_cmp)]
                        sm_cmp = (torch.softmax(pred_cmp["expression"], dim=1).cpu().numpy()[0]
                                if have_pred_cmp else np.zeros(len(names_cmp), dtype=np.float32))
                    else:
                        names_cmp, sm_cmp = None, None

                    prob_panel = draw_prob_barchart(
                        sm_main, names_main, sm_cmp, names_cmp, height=left_h
                    )

                    tmp = np.zeros((left_h, visualization.shape[1] + prob_panel.shape[1], 3), dtype=np.uint8)
                    tmp[:, :visualization.shape[1]] = visualization
                    tmp[:, visualization.shape[1]:] = prob_panel
                    visualization = tmp

                    
                


                # recording (fixed size)
                visualization = self._ensure_writer(visualization, snap["output_path"], fps=24.0)

                # publish to UI
                BUS.publish(cv2.cvtColor(visualization, cv2.COLOR_BGR2RGB))

                frame_count += 1
                if frame_count % 100 == 0:
                    log.info(f"Processed {frame_count} frames")

        finally:
            BUS.set_running(False)
            BUS.clear()
            cap.release()
            self._close_writer()
            cv2.destroyAllWindows()
            log.info("🎬 Capture loop stopped.")
