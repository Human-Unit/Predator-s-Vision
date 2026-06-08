"""
core/vision_engine.py
=====================
OpenCV vision-mode pipeline for the Yautja-Vision Bio-Mask HUD.

Each public function accepts a raw BGR NumPy frame and returns a processed
BGR frame of the same spatial dimensions.  All parameter sanitisation happens
here so callers (main.py) never have to guard against bad LLM values.

Dispatch is handled by apply_mode(), the single entry-point used by the run loops.
"""

import cv2
import numpy as np

from config.settings import DEFAULT_BLUR_STRENGTH, DEFAULT_ZOOM_SCALE, DEFAULT_SPECTRUM_TYPE


# ---------------------------------------------------------------------------
# THERMAL_VISION
# ---------------------------------------------------------------------------
def apply_thermal(frame: np.ndarray) -> np.ndarray:
    """
    Convert the frame to grayscale and map it through the JET colormap.

    cv2.COLORMAP_JET assigns:
      cold (low intensity)  → blue
      warm (mid intensity)  → yellow / green
      hot  (high intensity) → red

    This simulates an infrared heat-signature camera.
    """
    gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    thermal = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    return thermal


# ---------------------------------------------------------------------------
# TACTICAL_ZOOM
# ---------------------------------------------------------------------------
def apply_zoom(frame: np.ndarray, scale: float = DEFAULT_ZOOM_SCALE) -> np.ndarray:
    """
    Simulate a digital weapon-scope zoom by center-cropping then upscaling.
    Uses cv2.UMat for OpenCL-accelerated resizing if available.
    """
    try:
        scale = float(scale)
    except (TypeError, ValueError):
        scale = DEFAULT_ZOOM_SCALE
    scale = max(1.1, min(scale, 10.0))

    h, w = frame.shape[:2]
    crop_w = int(w / scale)
    crop_h = int(h / scale)

    cx, cy = w // 2, h // 2
    x1 = max(0, cx - crop_w // 2)
    y1 = max(0, cy - crop_h // 2)
    x2 = min(w, x1 + crop_w)
    y2 = min(h, y1 + crop_h)

    cropped = frame[y1:y2, x1:x2]

    # GPU Acceleration: Wrap in UMat for OpenCL resize
    umat = cv2.UMat(cropped)
    resized = cv2.resize(umat, (w, h), interpolation=cv2.INTER_LANCZOS4)
    return resized.get()


# ---------------------------------------------------------------------------
# TARGET_HUD
# ---------------------------------------------------------------------------
def apply_target_hud(frame: np.ndarray) -> np.ndarray:
    """
    Draw targeting crosshairs, focal rings, and corner lock-on brackets.

    All geometric primitives are drawn with cv2.LINE_AA for sub-pixel
    anti-aliasing, giving the reticle a sharp, crisp appearance even at
    lower resolutions.  The rotating outer ring and mode-specific notches
    are handled separately in ui/hud_overlays.py to keep this function
    stateless and frame-independent.
    """
    out  = frame.copy()
    h, w = out.shape[:2]
    cx, cy = w // 2, h // 2
    red  = (0, 0, 255)

    # Central dot + inner ring
    cv2.circle(out, (cx, cy), 14, red, 1, lineType=cv2.LINE_AA)
    cv2.circle(out, (cx, cy),  2, red, -1)

    # Crosshair gap lines (leave a 10-px gap around centre to avoid occlusion)
    gap = 10
    arm = 28
    cv2.line(out, (cx - arm - gap, cy), (cx - gap, cy), red, 1, cv2.LINE_AA)
    cv2.line(out, (cx + gap, cy), (cx + arm + gap, cy), red, 1, cv2.LINE_AA)
    cv2.line(out, (cx, cy - arm - gap), (cx, cy - gap), red, 1, cv2.LINE_AA)
    cv2.line(out, (cx, cy + gap), (cx, cy + arm + gap), red, 1, cv2.LINE_AA)

    # Corner lock-on brackets
    b, d = 18, 72
    for sx, sy in [(-1, -1), (1, -1), (-1, 1), (1, 1)]:
        ox, oy = cx + sx * d, cy + sy * d
        cv2.line(out, (ox, oy), (ox + sx * b, oy), red, 1)
        cv2.line(out, (ox, oy), (ox, oy + sy * b), red, 1)

    cv2.putText(out, "LOCK: DETECTED", (cx - 52, cy + 105),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, red, 1, cv2.LINE_AA)
    return out


# ---------------------------------------------------------------------------
# CLOAK_BLUR
# ---------------------------------------------------------------------------
def apply_cloak(frame: np.ndarray, strength: int = DEFAULT_BLUR_STRENGTH) -> np.ndarray:
    """
    Simulate active camouflage with OpenCL acceleration.
    """
    try:
        k = int(strength)
    except (TypeError, ValueError):
        k = DEFAULT_BLUR_STRENGTH
    k = max(3, min(k, 199))
    if k % 2 == 0:
        k += 1

    # GPU Acceleration: Wrap in UMat for heavy GaussianBlur and Laplacian
    umat = cv2.UMat(frame)
    blurred_umat = cv2.GaussianBlur(umat, (k, k), sigmaX=0)

    gray_umat  = cv2.cvtColor(umat, cv2.COLOR_BGR2GRAY)
    edges_umat = cv2.Laplacian(gray_umat, cv2.CV_8U, ksize=3)

    # Back to CPU for channel manipulation (OpenCV UMat support for slicing is limited)
    edges = edges_umat.get()
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    edges_bgr[:, :, 2] = 0

    return cv2.addWeighted(blurred_umat.get(), 0.88, edges_bgr, 0.12, gamma=0)


# ---------------------------------------------------------------------------
# NIGHT_VISION
# ---------------------------------------------------------------------------
def apply_night_vision(frame: np.ndarray) -> np.ndarray:
    """
    Simulate Yautja light-amplification mode (high-contrast blue/gray).

    Steps:
      1. Convert to grayscale.
      2. Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to
         amplify faint details without blowing out highlights.
      3. Tint the resulting map with a custom blue/gray BGR profile.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Light amplification via CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    amplified = clahe.apply(gray)

    # High-contrast Yautja Blue/Gray tinting
    # Mapping: B=0.9, G=0.65, R=0.5
    b = (amplified * 0.90).astype(np.uint8)
    g = (amplified * 0.65).astype(np.uint8)
    r = (amplified * 0.50).astype(np.uint8)

    return cv2.merge([b, g, r])


# ---------------------------------------------------------------------------
# SPECTRUM_SHIFT
# ---------------------------------------------------------------------------
def apply_spectrum(frame: np.ndarray, shift_type: str = DEFAULT_SPECTRUM_TYPE) -> np.ndarray:
    """
    Isolate alien visual frequencies by channel manipulation.

    shift_type options:
      "invert"     — bitwise NOT (full colour inversion)
      "red_only"   — zero B and G channels; keep R
      "green_only" — zero B and R channels; keep G
      "blue_only"  — zero G and R channels; keep B
    """
    st = str(shift_type).lower().strip()

    if st == "invert":
        return cv2.bitwise_not(frame)

    isolated = np.zeros_like(frame)
    channel_map = {"red_only": 2, "green_only": 1, "blue_only": 0}
    if st in channel_map:
        idx = channel_map[st]
        isolated[:, :, idx] = frame[:, :, idx]
        return isolated

    # Unknown shift_type — fall back to full inversion
    return cv2.bitwise_not(frame)


# ---------------------------------------------------------------------------
# Public dispatch function
# ---------------------------------------------------------------------------
def apply_mode(frame: np.ndarray, mode: str, params: dict) -> np.ndarray:
    """
    Single entry-point: route *mode* to the correct filter function.

    Args:
        frame:  Raw BGR input frame.
        mode:   One of the SUPPORTED_MODES strings from config/settings.py.
        params: Parameter dict extracted from the LLM JSON payload.

    Returns:
        Processed BGR frame of the same spatial dimensions as *frame*.
    """
    if mode == "THERMAL_VISION":
        return apply_thermal(frame)

    if mode == "TACTICAL_ZOOM":
        return apply_zoom(frame, scale=params.get("scale", DEFAULT_ZOOM_SCALE))

    if mode == "TARGET_HUD":
        return apply_target_hud(frame)

    if mode == "CLOAK_BLUR":
        return apply_cloak(frame, strength=params.get("strength", DEFAULT_BLUR_STRENGTH))

    if mode == "SPECTRUM_SHIFT":
        return apply_spectrum(frame, shift_type=params.get("shift_type", DEFAULT_SPECTRUM_TYPE))

    if mode == "NIGHT_VISION":
        return apply_night_vision(frame)

    # NORMAL_VISION or any unrecognised mode → pass-through
    return frame.copy()
