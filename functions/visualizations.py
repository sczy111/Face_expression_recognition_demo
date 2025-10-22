# visualizations.py
import numpy as np
import cv2
from collections import deque
import torch
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from functions.paths import APP_DIR, resource_path

# Progress bars for valence & arousal
def draw_progress_bars(
    val_main: float, aro_main: float,
    val_cmp: float | None, aro_cmp: float | None,
    height: int, width: int | None = None
) -> np.ndarray:
    """
    One panel, split into two horizontal halves:
      - Top half: Valence & Arousal bars for MAIN model
      - Bottom half: Valence & Arousal bars for COMPARE model (if None, draw empty)
    Bars are centered at 0 with a midline; left=negative, right=positive.
   """

    if width is None:
        width = height // 2

    panel = np.zeros((height, width, 3), dtype=np.uint8)
    margin_x = 20

    half_h = height // 2
    bar_h = max(18, half_h // 6)
    inner_gap = bar_h // 2
    start_y_top = (half_h - (2 * bar_h + inner_gap)) // 2
    start_y_bot = half_h + start_y_top

    def draw_one_bar(y, value, label, neg_color, pos_color):
        # background
        cv2.rectangle(panel, (margin_x, y), (width - margin_x, y + bar_h), (40, 40, 40), -1)
        # center line
        mid_x = (margin_x + width - margin_x) // 2
        cv2.line(panel, (mid_x, y), (mid_x, y + bar_h), (200, 200, 200), 2)

        # normalized half-width
        half_w = (width - 2 * margin_x) // 2
        if value >= 0:
            end_x = int(mid_x + min(1.0, value) * half_w)
            cv2.rectangle(panel, (mid_x, y), (end_x, y + bar_h), pos_color, -1)
        else:
            end_x = int(mid_x + max(-1.0, value) * half_w)
            cv2.rectangle(panel, (end_x, y), (mid_x, y + bar_h), neg_color, -1)

        cv2.putText(panel, f"{label}: {value:.2f}", (margin_x, y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

    # TOP HALF (MAIN)
    draw_one_bar(
        start_y_top, val_main,
        "Valence",
        neg_color=(0, 0, 255),    # red for negative
        pos_color=(0, 255, 0)     # green for positive
    )
    draw_one_bar(
        start_y_top + bar_h + inner_gap, aro_main,
        "Arousal",
        neg_color=(0, 255, 255),  # yellow for negative
        pos_color=(255, 0, 0)     # blue for positive
    )

    # divide halves
    cv2.line(panel, (0, half_h), (width, half_h), (80, 80, 80), 2)
    cv2.putText(panel, "Main", (margin_x, start_y_top - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(panel, "Compare", (margin_x, start_y_bot - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1, cv2.LINE_AA)

    # BOTTOM HALF
    if val_cmp is not None and aro_cmp is not None:
        draw_one_bar(
            start_y_bot, val_cmp,
            "Valence",
            neg_color=(100, 0, 255),   # purple-ish
            pos_color=(255, 165, 0)    # orange
        )
        draw_one_bar(
            start_y_bot + bar_h + inner_gap, aro_cmp,
            "Arousal",
            neg_color=(0, 128, 255),   # teal-ish
            pos_color=(255, 255, 0)    # yellow
        )
    else:
        # draw empty placeholders (keeps layout stable)
        cv2.rectangle(panel, (margin_x, start_y_bot), (width - margin_x, start_y_bot + bar_h), (40, 40, 40), -1)
        cv2.rectangle(panel, (margin_x, start_y_bot + bar_h + inner_gap), (width - margin_x, start_y_bot + 2*bar_h + inner_gap), (40, 40, 40), -1)

    return panel



# Probabilities bar chart
def draw_prob_barchart(
    probs_main: np.ndarray,
    names_main: list[str],
    probs_cmp: np.ndarray | None,
    names_cmp: list[str] | None,
    height: int = 256,
    width: int = 200
) -> np.ndarray:
    """
    Two probability charts stacked vertically:
      top  = main model (probs_main/names_main)
      bottom = compare model (probs_cmp/names_cmp). If None, draws empty area.
    Each half uses its own class list length and spacing.
    """
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    half = height // 2

    def draw_one_half(y0: int, probs: np.ndarray, names: list[str], title: str):
        cv2.putText(panel, title, (6, y0 + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220,220,220), 1, cv2.LINE_AA)
        if probs is None or names is None:
            return
        n = len(names)
        # +1 leaves some top padding like your original
        bar_h = max(10, (half - 10) // (n + 1))
        for i, (cls, prob) in enumerate(zip(names, probs)):
            y_top = y0 + 10 + i*bar_h + 6
            y_bot = y0 + (i+1)*bar_h + 6
            bar_len = int(float(prob) * (width - 60))
            cv2.rectangle(panel, (50, y_top), (50 + bar_len, y_bot), (100,200,250), -1)
            cv2.putText(panel, f"{cls} {float(prob):.2f}", (5, y_bot - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

    # top (main)
    draw_one_half(0, probs_main, names_main, "")
    # divider
    cv2.line(panel, (0, half), (width, half), (80,80,80), 2)
    # bottom (compare)
    if probs_cmp is not None and names_cmp is not None:
        draw_one_half(half, probs_cmp, names_cmp, "")
    else:
        cv2.putText(panel, "", (6, half + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120,120,120), 1, cv2.LINE_AA)
    return panel




def draw_bbox_and_label(frame_rgb, face_bbox, emotion_prediction, emotion_classes):
    """Draw a bounding box and categorical emotion on the full frame."""
    frame_out = frame_rgb.copy()
    cv2.rectangle(frame_out,
                  (int(face_bbox[0]), int(face_bbox[1])),
                  (int(face_bbox[2]), int(face_bbox[3])),
                  (255, 0, 0), 2)
    predicted_emotion_class_idx = int(
        torch.argmax(torch.nn.functional.softmax(emotion_prediction["expression"], dim=1)).cpu().item()
    )
    cv2.putText(frame_out,
                emotion_classes.get(predicted_emotion_class_idx, "Unknown"),
                (int(face_bbox[0]), int(face_bbox[1]) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    return frame_out

def draw_bbox(frame_rgb, face_bbox):
    """Draw a bounding box and categorical emotion on the full frame."""
    frame_out = frame_rgb.copy()
    cv2.rectangle(frame_out,
                  (int(face_bbox[0]), int(face_bbox[1])),
                  (int(face_bbox[2]), int(face_bbox[3])),
                  (255, 0, 0), 2)
    return frame_out


def draw_face_landmarks(face_crop_rgb, emotion_prediction):
    """Return the cropped face with landmarks overlaid."""
    heatmap = torch.nn.functional.interpolate(
        emotion_prediction["heatmap"],
        (face_crop_rgb.shape[0], face_crop_rgb.shape[1]),
        mode="bilinear",
    )
    out = face_crop_rgb.copy()
    for landmark_idx in range(heatmap[0].shape[0]):
        channel = heatmap[0, landmark_idx, :, :]
        idx = torch.argmax(channel)
        y = (idx // channel.shape[1]).item()
        x = (idx % channel.shape[1]).item()
        cv2.circle(out, (int(x), int(y)), 2, (255, 255, 255), -1)
    return out


def draw_valence_arousal_circumplex(valence, arousal, size):
    """Return circumplex image with dot drawn at valence/arousal coordinates."""
    circumplex_path = Path(__file__).parent / "images/circumplex.png"
    img = cv2.imread(str(circumplex_path))
    img = cv2.resize(img, (size, size))
    pos = ((valence + 1.0) / 2.0 * size, (1.0 - arousal) / 2.0 * size)
    cv2.circle(img, (int(pos[0]), int(pos[1])), 10, (0, 0, 255), -1)
    return img

# Draw two progression bars around the box (test only, not actually used in this project)
def draw_bars_around_bbox(frame_bgr, bbox, valence, arousal):
    """
    Draw valence (horizontal) and arousal (vertical) bars anchored to the bbox,
    with semantic labels and axis titles. No face box is drawn.
    """
    

    x1, y1, x2, y2 = [int(v) for v in bbox]
    h, w = frame_bgr.shape[:2]
    out = frame_bgr.copy()

    # Clamp bbox
    x1 = max(0, min(x1, w-1)); x2 = max(0, min(x2, w-1))
    y1 = max(0, min(y1, h-1)); y2 = max(0, min(y2, h-1))
    if x2 <= x1 or y2 <= y1:
        return out

    bw = x2 - x1
    bh = y2 - y1

    font_scale = 0.35
    title_scale = 0.5
    thickness = 1

    # Valence bar (horizontal) 
    bar_h = max(10, bh // 14)
    margin = max(6, bh // 30)
    vy1 = y2 + margin
    vy2 = vy1 + bar_h
    if vy2 >= h:
        vy2 = y1 - margin
        vy1 = vy2 - bar_h
    if vy1 < 0:
        vy1 = y2 - bar_h - 2
        vy2 = y2 - 2
    vx1, vx2 = x1, x2

    cv2.rectangle(out, (vx1, vy1), (vx2, vy2), (40,40,40), -1)
    mid_x = (vx1 + vx2) // 2
    cv2.line(out, (mid_x, vy1), (mid_x, vy2), (200,200,200), 1)

    v = float(np.clip(valence, -1.0, 1.0))
    px = int((v + 1.0) * 0.5 * (vx2 - vx1) + vx1); px = int(np.clip(px, vx1, vx2))
    if v >= 0: cv2.rectangle(out, (mid_x, vy1), (px, vy2), (0,200,0), -1)   # green
    else:      cv2.rectangle(out, (px,   vy1), (mid_x, vy2), (0,0,255), -1) # red

    # Valence labels (below the bar)
    font      = cv2.FONT_HERSHEY_SIMPLEX
    f_scale   = 0.35
    thickness = 1

    # Baseline Y: prefer below the bar, else above
    gap = max(10, bar_h // 3)
    val_lab_y = vy2 + gap
    if val_lab_y + 4 > h:                 # not enough room below
        val_lab_y = max(12, vy1 - 6)      # draw above the bar

    # Text widths/heights (tuples)
    neg, neu, pos = "negative", "neutral", "positive"
    neg_sz = cv2.getTextSize(neg, font, f_scale, thickness)[0]  # (w,h)
    neu_sz = cv2.getTextSize(neu, font, f_scale, thickness)[0]
    pos_sz = cv2.getTextSize(pos, font, f_scale, thickness)[0]
    neg_w, neu_w, pos_w = neg_sz[0], neu_sz[0], pos_sz[0]

    # X positions: left-aligned, centered, right-aligned
    x_neg = max(0, vx1)                                      # left edge
    x_neu = max(0, min(w - neu_w, mid_x - neu_w // 2))       # centered
    x_pos = max(0, vx2 - pos_w)                               # right edge

    cv2.putText(out, neg, (x_neg, val_lab_y), font, f_scale, (0,  0,255), thickness, cv2.LINE_AA)
    cv2.putText(out, neu, (x_neu, val_lab_y), font, f_scale, (255,255,255), thickness, cv2.LINE_AA)
    cv2.putText(out, pos, (x_pos, val_lab_y), font, f_scale, (0,200,  0), thickness, cv2.LINE_AA)

    # Axis title, centered
    title = "Valence level"
    title_scale = 0.5
    title_sz = cv2.getTextSize(title, font, title_scale, thickness)[0]  # (w,h)
    title_w, title_h = title_sz[0], title_sz[1]
    title_y = val_lab_y + title_h + 6
    if title_y > h - 2:
        title_y = max(12, vy1 - 10)
    title_x = max(0, min(w - title_w, mid_x - title_w // 2))
    cv2.putText(out, title, (title_x, title_y), font, title_scale, (255,255,255), thickness, cv2.LINE_AA)




    # Arousal bar (vertical; LEFT of face preferred)
    bar_w = max(10, bw // 20)
    margin = max(6, bw // 30)
    ax2 = x1 - margin
    ax1 = ax2 - bar_w
    if ax1 < 0:            # fallback: right of face
        ax1 = x2 + margin
        ax2 = ax1 + bar_w
    if ax2 > w:            # fallback: inside right edge
        ax2 = x2
        ax1 = ax2 - bar_w
    ay1, ay2 = y1, y2

    cv2.rectangle(out, (ax1, ay1), (ax2, ay2), (40,40,40), -1)
    mid_y = (ay1 + ay2) // 2
    cv2.line(out, (ax1, mid_y), (ax2, mid_y), (200,200,200), 1)

    a = float(np.clip(arousal, -1.0, 1.0))
    py = int((1.0 - (a + 1.0) * 0.5) * (ay2 - ay1) + ay1); py = int(np.clip(py, ay1, ay2))
    if a >= 0: cv2.rectangle(out, (ax1, py), (ax2, mid_y), (255,0,0), -1)      # blue (excited)
    else:      cv2.rectangle(out, (ax1, mid_y), (ax2, py), (0,220,180), -1)    # yellow-green (calm)

    # Arousal labels on the LEFT of the bar (right-aligned to bar edge)
    
    pad = 6
    right_x = ax1 - pad  # right edge for left-side labels

    font = cv2.FONT_HERSHEY_SIMPLEX
    f_scale = 0.45
    thickness = 1

    # sizes for alignment
    exc_sz = cv2.getTextSize("exciting", font, f_scale, thickness)[0]  # (w,h)
    neu_sz = cv2.getTextSize("neutral",  font, f_scale, thickness)[0]
    cal_sz = cv2.getTextSize("calming",  font, f_scale, thickness)[0]

    # Draw "exciting" (top-left)
    x_exc = max(0, right_x - exc_sz[0])
    y_exc = max(exc_sz[1] + 2, ay1 + exc_sz[1])
    cv2.putText(out, "exciting", (x_exc, y_exc), font, f_scale, (255, 0, 0), thickness, cv2.LINE_AA)

    # Draw "calming" (bottom-left)
    x_cal = max(0, right_x - cal_sz[0])
    y_cal = min(h - 2, ay2)  # baseline at bottom of bar
    cv2.putText(out, "calming", (x_cal, y_cal), font, f_scale, (0, 220, 180), thickness, cv2.LINE_AA)

    # Draw "neutral" at middle-left (we also keep its exact position to place the title)
    x_neu = max(0, right_x - neu_sz[0])
    # place its baseline at the midline
    y_neu = int(np.clip(mid_y + neu_sz[1] // 2, neu_sz[1] + 2, h - 2))
    cv2.putText(out, "neutral", (x_neu, y_neu), font, f_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    # Two-line title "Arousal level" centered as a block
    title1, title2 = "Arousal", "level"
    t1_sz = cv2.getTextSize(title1, font, 0.5, thickness)[0]  # (w,h)
    t2_sz = cv2.getTextSize(title2, font, 0.5, thickness)[0]

    gap_x = 8  # horizontal gap between title and "neutral"
    title_right = max(0, x_neu - gap_x)
    block_w = max(t1_sz[0], t2_sz[0])

    # center each line inside the block width
    x_title1 = title_right - block_w + (block_w - t1_sz[0]) // 2
    x_title2 = title_right - block_w + (block_w - t2_sz[0]) // 2

    # vertically center two lines around y_neu
    line_gap = 2
    block_h = t1_sz[1] + line_gap + t2_sz[1]
    y_title1 = int(np.clip(y_neu - block_h // 2 + t1_sz[1], t1_sz[1] + 2, h - 2))
    y_title2 = int(np.clip(y_title1 + t2_sz[1] + line_gap, t2_sz[1] + 2, h - 2))

    cv2.putText(out, title1, (x_title1, y_title1), font, 0.5, (255, 255, 255), thickness, cv2.LINE_AA)
    cv2.putText(out, title2, (x_title2, y_title2), font, 0.5, (255, 255, 255), thickness, cv2.LINE_AA)



    return out


# draw the emotion label("Happy", "neutral" etc.)
def draw_label_only(frame_bgr, bbox, emotion_prediction, emotion_classes):
    """
    Writes the categorical emotion text near the bbox center-top.
    No rectangle is drawn.
    """
    import cv2, torch
    x1, y1, x2, y2 = [int(v) for v in bbox]
    cx = (x1 + x2) // 2
    ty = max(15, y1 - 10)

    idx = int(torch.argmax(
        torch.nn.functional.softmax(emotion_prediction["expression"], dim=1)
    ).cpu().item())

    cx = cx -50
    label = emotion_classes.get(idx, "Unknown")
    out = frame_bgr.copy()
    cv2.putText(out, label, (cx, ty),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return out





_EMOJI_FONT_CANDIDATES = [
    str(resource_path("functions", "seguiemj.ttf"))
]

def _find_emoji_font(size: int):
    for p in _EMOJI_FONT_CANDIDATES:
        if Path(p).exists():
            try:
                return ImageFont.truetype(p, size=size)
            except Exception:
                pass
    return None

_EMOJI_MAP = {
    "neutral":  "😐",
    "happy":    "😄",
    "sad":      "😢",
    "surprise": "😮",
    "fear":     "😱",
    "disgust":  "🤢",
    "anger":    "😠",
    "contempt": "😒",
}
def _render_emoji_rgba(label_text: str, target_h: int = 36, pad: int = 4) -> np.ndarray | None:
    """
    Render a color emoji (RGBA) using a color-emoji capable font.
    Uses a large temp canvas + textbbox so no part of the emoji gets cropped.
    """
    ch = _EMOJI_MAP.get(label_text.lower())
    if not ch:
        return None

    font = _find_emoji_font(size=target_h * 2)  # draw bigger to keep detail
    if font is None:
        return None

    # big transparent canvas to avoid clipping
    C = target_h * 4
    img = Image.new("RGBA", (C, C), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # put roughly in the middle, then measure exact bbox
    center = (C // 2, C // 2)
    draw.text(center, ch, font=font, embedded_color=True, anchor="mm")
    bbox = draw.textbbox(center, ch, font=font, anchor="mm")

    # crop to glyph bbox with a little padding
    x0 = max(0, bbox[0] - pad)
    y0 = max(0, bbox[1] - pad)
    x1 = min(C, bbox[2] + pad)
    y1 = min(C, bbox[3] + pad)
    cropped = img.crop((x0, y0, x1, y1))

    w, h = cropped.size
    if h != target_h:
        new_w = max(1, int(round(w * (target_h / h))))
        cropped = cropped.resize((new_w, target_h), Image.LANCZOS)

    return np.array(cropped)  # RGBA


def _paste_rgba_on_bgr(dst_bgr: np.ndarray, src_rgba: np.ndarray, x: int, y: int) -> None:
    h, w = src_rgba.shape[:2]
    H, W = dst_bgr.shape[:2]
    if x >= W or y >= H:
        return
    x2 = min(x + w, W); y2 = min(y + h, H)
    w = x2 - x; h = y2 - y
    if w <= 0 or h <= 0:
        return
    src = src_rgba[:h, :w]
    bgr = dst_bgr[y:y+h, x:x+w]
    alpha = src[:, :, 3:4] / 255.0
    rgb = src[:, :, :3][:, :, ::-1]  # RGBA->BGR
    bgr[:] = (alpha * rgb + (1.0 - alpha) * bgr).astype(np.uint8)



def draw_emoji_right_of_label(
    frame_bgr, bbox, emotion_prediction, emotion_classes,
    font_scale: float = 0.8, icon_h: int = 28,
    fixed_offset_x: int = 120, fixed_offset_y: int = -10
):

    # Get emotion label
    x1, y1, x2, y2 = [int(v) for v in bbox]
    cx = (x1 + x2) // 2
    ty = max(15, y1 - 10)

    idx = int(torch.argmax(
        torch.nn.functional.softmax(emotion_prediction["expression"], dim=1)
    ).cpu().item())
    label = emotion_classes.get(idx, "Unknown")

    # Render emoji
    icon = _render_emoji_rgba(label, target_h=icon_h)
    if icon is None:
        return frame_bgr

    ih, iw = icon.shape[:2]
    H, W = frame_bgr.shape[:2]

    # Fixed placement
    ex = int(cx + fixed_offset_x)
    ey = int(ty + fixed_offset_y)

    # Clamp inside frame
    ex = max(0, min(ex, W - iw))
    ey = max(0, min(ey, H - ih))

    out = frame_bgr.copy()
    _paste_rgba_on_bgr(out, icon, ex, ey)
    return out




# Separate the function "draw_bars_around_bbox" above to two functions, one draws the bar with only titles, one draws the labels.
def draw_bars_with_titles_only(frame_bgr, bbox, valence, arousal):
    """
    Same bars/placement as draw_bars_around_bbox, but ONLY draws the titles:
      - "Valence level" centered under the horizontal bar
      - "Arousal" / "level" (two lines) to the LEFT of the vertical bar,
        vertically centered on the neutral baseline.
    No face box or other labels are drawn.
    """
    import cv2, numpy as np

    x1, y1, x2, y2 = [int(v) for v in bbox]
    h, w = frame_bgr.shape[:2]
    out = frame_bgr.copy()

    # clamp bbox
    x1 = max(0, min(x1, w-1)); x2 = max(0, min(x2, w-1))
    y1 = max(0, min(y1, h-1)); y2 = max(0, min(y2, h-1))
    if x2 <= x1 or y2 <= y1:
        return out

    bw = x2 - x1
    bh = y2 - y1

    # Valence bar (horizontal)
    bar_h = max(10, bh // 14)
    margin = max(6, bh // 30)
    vy1 = y2 + margin
    vy2 = vy1 + bar_h
    if vy2 >= h:
        vy2 = y1 - margin
        vy1 = vy2 - bar_h
    if vy1 < 0:
        vy1 = y2 - bar_h - 2
        vy2 = y2 - 2
    vx1, vx2 = x1, x2

    cv2.rectangle(out, (vx1, vy1), (vx2, vy2), (40,40,40), -1)
    mid_x = (vx1 + vx2) // 2
    cv2.line(out, (mid_x, vy1), (mid_x, vy2), (200,200,200), 1)

    v = float(np.clip(valence, -1.0, 1.0))
    px = int((v + 1.0) * 0.5 * (vx2 - vx1) + vx1); px = int(np.clip(px, vx1, vx2))
    if v >= 0: cv2.rectangle(out, (mid_x, vy1), (px, vy2), (0,200,0), -1)   # green
    else:      cv2.rectangle(out, (px,   vy1), (mid_x, vy2), (0,0,255), -1) # red

    # Title under valence bar
    font = cv2.FONT_HERSHEY_SIMPLEX
    title = "Valence level"
    title_scale = 0.5
    thickness = 1
    title_sz = cv2.getTextSize(title, font, title_scale, thickness)[0]  # (w,h)
    title_w, title_h = title_sz[0], title_sz[1]
    extra_gap = 18 # adjust this gap as you need
    title_y = vy2 + title_h + extra_gap
    if title_y > h - 2:
        title_y = max(12, vy1 - 10)
    title_x = max(0, min(w - title_w, mid_x - title_w // 2))
    cv2.putText(out, title, (title_x, title_y), font, title_scale, (255,255,255), thickness, cv2.LINE_AA)

    # Arousal bar (vertical; LEFT of face preferred)
    bar_w = max(10, bw // 20)
    margin = max(6, bw // 30)
    ax2 = x1 - margin
    ax1 = ax2 - bar_w
    if ax1 < 0:
        ax1 = x2 + margin
        ax2 = ax1 + bar_w
    if ax2 > w:
        ax2 = x2
        ax1 = ax2 - bar_w
    ay1, ay2 = y1, y2

    cv2.rectangle(out, (ax1, ay1), (ax2, ay2), (40,40,40), -1)
    mid_y = (ay1 + ay2) // 2
    cv2.line(out, (ax1, mid_y), (ax2, mid_y), (200,200,200), 1)

    a = float(np.clip(arousal, -1.0, 1.0))
    py = int((1.0 - (a + 1.0) * 0.5) * (ay2 - ay1) + ay1); py = int(np.clip(py, ay1, ay2))
    if a >= 0: cv2.rectangle(out, (ax1, py), (ax2, mid_y), (255,0,0), -1)      # blue (excited)
    else:      cv2.rectangle(out, (ax1, mid_y), (ax2, py), (0,220,180), -1)    # yellow-green (calm)

    # Two-line title to LEFT of neutral baseline
    title1, title2 = "Arousal", "level"
    t1_sz = cv2.getTextSize(title1, font, 0.5, thickness)[0]  # (w,h)
    t2_sz = cv2.getTextSize(title2, font, 0.5, thickness)[0]
    line_gap = 2
    block_w = max(t1_sz[0], t2_sz[0])
    block_h = t1_sz[1] + line_gap + t2_sz[1]

    # Right edge of the two-line block stays a little left of the bar
    pad_left = 8
    title_right = max(0, ax1 - pad_left)
    x_title1 = title_right - block_w + (block_w - t1_sz[0]) // 2
    x_title2 = title_right - block_w + (block_w - t2_sz[0]) // 2

    y_title1 = int(np.clip(mid_y - block_h // 2 + t1_sz[1], t1_sz[1] + 2, h - 2))
    y_title2 = int(np.clip(y_title1 + t2_sz[1] + line_gap, t2_sz[1] + 2, h - 2))
    x_title1 -= 40   
    x_title2 -= 40   # push the title left, adjust as you need

    cv2.putText(out, title1, (x_title1, y_title1), font, 0.5, (255,255,255), thickness, cv2.LINE_AA)
    cv2.putText(out, title2, (x_title2, y_title2), font, 0.5, (255,255,255), thickness, cv2.LINE_AA)

    return out

# Draws only the bar labels
def draw_bars_labels_only(frame_bgr, bbox):
    """
    Draw ONLY the semantic labels around the bars (no bars, no titles):
      - Valence: 'negative' | 'neutral' | 'positive' under/above the horizontal bar
      - Arousal: 'exciting' | 'neutral' | 'calming' to the LEFT of the vertical bar
    Geometry matches draw_bars_titles_only / draw_bars_around_bbox so the two can be composed.
    """
    import cv2, numpy as np

    x1, y1, x2, y2 = [int(v) for v in bbox]
    h, w = frame_bgr.shape[:2]
    out = frame_bgr.copy()

    # Clamp bbox
    x1 = max(0, min(x1, w - 1)); x2 = max(0, min(x2, w - 1))
    y1 = max(0, min(y1, h - 1)); y2 = max(0, min(y2, h - 1))
    if x2 <= x1 or y2 <= y1:
        return out

    bw = x2 - x1
    bh = y2 - y1

    # Valence bar geometry (same as bars_titles_only)
    bar_h = max(10, bh // 14)
    margin = max(6, bh // 30)
    vy1 = y2 + margin
    vy2 = vy1 + bar_h
    if vy2 >= h:
        vy2 = y1 - margin
        vy1 = vy2 - bar_h
    if vy1 < 0:
        vy1 = y2 - bar_h - 2
        vy2 = y2 - 2
    vx1, vx2 = x1, x2
    mid_x = (vx1 + vx2) // 2

    # Valence labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    f_scale = 0.35 # adjust the label font size as you need
    thickness = 1

    gap = max(10, bar_h // 3)
    val_lab_y = vy2 + gap
    if val_lab_y + 4 > h:
        val_lab_y = max(12, vy1 - 6)

    neg, neu, pos = "negative", "neutral", "positive"
    neg_sz = cv2.getTextSize(neg, font, f_scale, thickness)[0]
    neu_sz = cv2.getTextSize(neu, font, f_scale, thickness)[0]
    pos_sz = cv2.getTextSize(pos, font, f_scale, thickness)[0]
    neg_w, neu_w, pos_w = neg_sz[0], neu_sz[0], pos_sz[0]

    x_neg = max(0, vx1)
    x_neu = max(0, min(w - neu_w, mid_x - neu_w // 2))
    x_pos = max(0, vx2 - pos_w)

    cv2.putText(out, neg, (x_neg, val_lab_y), font, f_scale, (50,  50,255), thickness, cv2.LINE_AA)
    cv2.putText(out, neu, (x_neu, val_lab_y), font, f_scale, (255,255,255), thickness, cv2.LINE_AA)
    cv2.putText(out, pos, (x_pos, val_lab_y), font, f_scale, (50,200,  50), thickness, cv2.LINE_AA)

    #  Arousal bar geometry
    bar_w = max(10, bw // 20)
    margin = max(6, bw // 30)
    ax2 = x1 - margin
    ax1 = ax2 - bar_w
    if ax1 < 0:            # fallback: right of face
        ax1 = x2 + margin
        ax2 = ax1 + bar_w
    if ax2 > w:            # fallback: inside right edge
        ax2 = x2
        ax1 = ax2 - bar_w
    ay1, ay2 = y1, y2
    mid_y = (ay1 + ay2) // 2

    # Arousal labels on the LEFT
    pad = 6
    right_x = ax1 - pad  # right-aligned to bar's left edge

    exc_sz = cv2.getTextSize("exciting", font, f_scale, thickness)[0]
    neu_sz = cv2.getTextSize("neutral",  font, f_scale, thickness)[0]
    cal_sz = cv2.getTextSize("calming",  font, f_scale, thickness)[0]

    x_exc = max(0, right_x - exc_sz[0])
    y_exc = max(exc_sz[1] + 2, ay1 + exc_sz[1])
    cv2.putText(out, "exciting", (x_exc, y_exc), font, f_scale, (255, 150, 0), thickness, cv2.LINE_AA)

    x_cal = max(0, right_x - cal_sz[0])
    y_cal = min(h - 2, ay2)
    cv2.putText(out, "calming", (x_cal, y_cal), font, f_scale, (0, 220, 180), thickness, cv2.LINE_AA)

    x_neu = max(0, right_x - neu_sz[0])
    y_neu = int(np.clip(mid_y + neu_sz[1] // 2, neu_sz[1] + 2, h - 2))
    cv2.putText(out, "neutral", (x_neu, y_neu), font, f_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    return out


def draw_valence_arousal_circumplex_from_vals(
    valence: float,
    arousal: float,
    panel_size: int = 512,
    image_name: str = "circumplex.png",
    point_radius: int = 16,
    point_bgr: tuple = (0, 0, 255),  # red dot in BGR
):
    """
    Draw a (valence, arousal) point on a circumplex background.
    Expects valence/arousal in [-1, 1]. Returns a BGR uint8 panel of shape (panel_size, panel_size, 3).
    """
    # Resolve the image path relative to this file
    img_path = resource_path("functions", "images", image_name)
    circumplex = cv2.imread(str(img_path))
    if circumplex is None:
        # fallback to blank panel if image missing
        circumplex = np.zeros((panel_size, panel_size, 3), dtype=np.uint8)
    else:
        circumplex = cv2.resize(circumplex, (panel_size, panel_size))

    # map valence/arousal [-1,1] -> [0, panel_size]
    x = int(((float(valence) + 1.0) / 2.0) * panel_size)
    y = int(((1.0 - float(arousal)) / 2.0) * panel_size)  # arousal up

    cv2.circle(circumplex, (x, y), point_radius, point_bgr, -1)
    return circumplex

