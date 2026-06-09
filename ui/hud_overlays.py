"""
ui/hud_overlays.py
==================
Cinematic Yautja Bio-Mask HUD — upgraded visual design.

Public surface:
  draw_hud(frame, mode, error_state, routing_active)  → frame
  apply_lens(frame)                                   → frame
  apply_glitch(frame)                                 → frame

Design upgrades over v1:
  • Hexagonal mesh background layer
  • Multi-ring rotating reticle (3 concentric rings)
  • Distance/threat bar gauges on both sides
  • Target acquisition boxes (TARGET_HUD / TACTICAL_ZOOM)
  • Richer telemetry panel with animated signal bar
  • Corner brackets with measurement tick marks
  • Mode-aware colour palette (thermal / target / cloak / tactical)
  • Subtle chromatic aberration (1 px split — barely perceptible)
  • Soft edge-only vignette (centre is 100 % brightness, gentle falloff)
  • No CRT scanlines — full image clarity preserved
  • Glitch adds vertical channel tears in addition to row rolls
"""

import time
import random
import math
import cv2
import numpy as np

from config.settings import (
    COLOR_GREEN, COLOR_RED, COLOR_CYAN,
    FONT, FONT_SCALE_SM, FONT_SCALE_MD, FONT_SCALE_LG,
    FONT_THICKNESS, FONT_THICKNESS_BOLD,
    APP_NAME, APP_VERSION,
)


# ---------------------------------------------------------------------------
# Palette — one entry per visor mode
# ---------------------------------------------------------------------------
_PALETTES = {
    "THERMAL":       {"primary": (0,  100, 255), "secondary": (0,  180, 255), "accent": (0,   40, 220)},
    "TARGET_HUD":    {"primary": (50, 220,  50), "secondary": (200, 220,  0), "accent": (40,  40, 220)},
    "CLOAK_BLUR":    {"primary": (220, 160,  0), "secondary": (255, 210, 60), "accent": (160,  80, 255)},
    "TACTICAL_ZOOM": {"primary": (0,  220, 180), "secondary": (0,  255, 120), "accent": (0,  140, 255)},
    "NIGHT_VISION":  {"primary": (255, 180,  0), "secondary": (255, 100, 50), "accent": (255, 255, 255)},
    "_default":      {"primary": (60, 220,  60), "secondary": (0,  200, 100), "accent": (60,  60, 200)},
}

def _palette(mode: str) -> dict:
    return _PALETTES.get(mode, _PALETTES["_default"])


# ---------------------------------------------------------------------------
# Internal caches
# ---------------------------------------------------------------------------
_VIGNETTE_CACHE: dict[tuple[int, int], np.ndarray] = {}
_HEX_CACHE:      dict[tuple[int, int, str], np.ndarray] = {}



# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _hud_color(mode: str, error: bool) -> tuple[int, int, int]:
    if error:
        return COLOR_RED
    return _palette(mode)["primary"]


def _get_vignette_mask(h: int, w: int) -> np.ndarray:
    """
    Cached 3-channel float32 [0..1] radial vignette mask.
    Wide sigma keeps centre at 1.0 with gentle corner falloff.
    """
    key = (h, w)
    if key not in _VIGNETTE_CACHE:
        kx   = cv2.getGaussianKernel(w, w * 0.65)
        ky   = cv2.getGaussianKernel(h, h * 0.65)
        mask = ky * kx.T
        mask = mask / mask.max()
        # centre = 1.0, corners ≈ 0.72
        v2d  = (0.72 + 0.28 * mask).astype(np.float32)
        # Cache as BGR to avoid broadcasting in every frame
        _VIGNETTE_CACHE[key] = cv2.merge([v2d, v2d, v2d])
    return _VIGNETTE_CACHE[key]


def _get_hex_overlay(h: int, w: int, color_key: str) -> np.ndarray:
    """
    Cached BGR hex-grid layer (uint8, same size as frame).
    Hex cell radius ≈ 22 px; drawn with alpha 0.07 so it
    stays subtle behind the main imagery.
    """
    key = (h, w, color_key)
    if key not in _HEX_CACHE:
        p      = _PALETTES.get(color_key, _PALETTES["_default"])
        col    = p["primary"]
        layer  = np.zeros((h, w, 3), dtype=np.uint8)
        sz     = 22
        dx     = int(sz * 1.732)
        dy     = int(sz * 1.5)
        for row in range(-1, h // dy + 3):
            for col_i in range(-1, w // dx + 3):
                ox = (row % 2) * (dx // 2)
                hx = col_i * dx + ox
                hy = row * dy
                pts = []
                for i in range(6):
                    ang = math.pi / 3 * i - math.pi / 6
                    pts.append((int(hx + sz * math.cos(ang)),
                                int(hy + sz * math.sin(ang))))
                pts_np = np.array(pts, dtype=np.int32)
                cv2.polylines(layer, [pts_np], True, col, 1, cv2.LINE_AA)
        _HEX_CACHE[key] = layer
    return _HEX_CACHE[key]


# ---------------------------------------------------------------------------
# Lens post-processing
# ---------------------------------------------------------------------------
def apply_lens(frame: np.ndarray) -> np.ndarray:
    """
    Optimised lens post-processing:
    1. Chromatic Aberration (1px split)
    2. Vignette (gentle corner falloff)
    """
    h, w = frame.shape[:2]

    # Optimized Chromatic Aberration: manual assignment to avoid dstack overhead
    aberrated = np.empty_like(frame)
    aberrated[:, :, 0] = np.roll(frame[:, :, 0], -1, axis=1)
    aberrated[:, :, 1] = frame[:, :, 1]
    aberrated[:, :, 2] = np.roll(frame[:, :, 2],  1, axis=1)

    # Optimized Vignette: use cv2.multiply with 3-channel cached float mask
    mask = _get_vignette_mask(h, w)
    return cv2.multiply(aberrated, mask, dtype=cv2.CV_8U)


# ---------------------------------------------------------------------------
# Glitch effect
# ---------------------------------------------------------------------------
def apply_glitch(frame: np.ndarray) -> np.ndarray:
    """
    Horizontal row-rolls + EMI static lines + vertical channel tears.
    Vertical tears (new): one BGR channel is rolled up/down independently,
    creating the coloured column streaks seen on damaged CRT screens.
    """
    h, w     = frame.shape[:2]
    glitched = frame.copy()

    # Horizontal pixel-row rolls
    for _ in range(random.randint(4, 10)):
        y  = random.randint(0, h - 30)
        sh = random.randint(3, 18)
        dx = random.randint(-80, 80)
        glitched[y:y + sh, :] = np.roll(glitched[y:y + sh, :], dx, axis=1)

    # EMI static lines
    for _ in range(random.randint(3, 7)):
        y  = random.randint(0, h - 4)
        sh = random.randint(1, 3)
        c  = (random.randint(80, 255), random.randint(80, 255), random.randint(80, 255))
        glitched[y:y + sh, :] = c

    # Vertical channel tear (new)
    ch  = random.randint(0, 2)
    dy  = random.randint(-30, 30)
    glitched[:, :, ch] = np.roll(glitched[:, :, ch], dy, axis=0)

    return glitched


# ---------------------------------------------------------------------------
# Sub-drawing routines
# ---------------------------------------------------------------------------

def _draw_hex_mesh(out: np.ndarray, mode: str, alpha: float = 0.07) -> None:
    """Blend the cached hexagonal grid into the frame."""
    h, w  = out.shape[:2]
    layer = _get_hex_overlay(h, w, mode)
    cv2.addWeighted(layer, alpha, out, 1.0, 0, dst=out)


def _draw_corner_brackets(out: np.ndarray, col: tuple, h: int, w: int) -> None:
    """
    Four L-shaped corner markers with 3 tick marks along each arm.
    Arm length scales with the shorter frame dimension.
    """
    pad = 18
    arm = int(min(h, w) * 0.055)
    for sx, sy in [(-1, -1), (1, -1), (-1, 1), (1, 1)]:
        bx = w - pad if sx == 1 else pad
        by = h - pad if sy == 1 else pad
        cv2.line(out, (bx,          by), (bx + sx * arm, by),          col, 2, cv2.LINE_AA)
        cv2.line(out, (bx,          by), (bx,            by + sy * arm), col, 2, cv2.LINE_AA)
        # tick marks along horizontal arm
        for i in range(1, 4):
            tx = bx + sx * arm * i // 4
            cv2.line(out, (tx, by - sy * 4), (tx, by + sy * 4), col, 1, cv2.LINE_AA)
        # tick marks along vertical arm
        for i in range(1, 4):
            ty = by + sy * arm * i // 4
            cv2.line(out, (bx - sx * 4, ty), (bx + sx * 4, ty), col, 1, cv2.LINE_AA)


def _draw_reticle(out: np.ndarray, P: dict, cx: int, cy: int, t: float) -> None:
    """
    Three-ring rotating reticle:
      • Outer ring   — dashed, slow CW rotation
      • Middle ring  — dashed, faster CCW rotation
      • Inner ring   — solid, static
      • 24-tick outer ring + crosshair gap + centre dot
    """
    r_out = int(min(out.shape[:2]) * 0.09)
    r_mid = int(r_out * 0.75)
    r_in  = int(r_out * 0.42)
    col_p = P["primary"]
    col_s = P["secondary"]
    col_a = P["accent"]

    # --- Outer dashed ring (CW) ---
    num_dashes = 32
    for i in range(num_dashes):
        if i % 3 == 0:                      # skip every 3rd segment → dashed look
            continue
        a1 = (i / num_dashes) * 2 * math.pi + t * 0.5
        a2 = ((i + 0.8) / num_dashes) * 2 * math.pi + t * 0.5
        p1 = (int(cx + r_out * math.cos(a1)), int(cy + r_out * math.sin(a1)))
        p2 = (int(cx + r_out * math.cos(a2)), int(cy + r_out * math.sin(a2)))
        cv2.line(out, p1, p2, col_p, 1, cv2.LINE_AA)

    # --- Middle dashed ring (CCW) ---
    for i in range(num_dashes):
        if i % 4 == 0:
            continue
        a1 = (i / num_dashes) * 2 * math.pi - t * 1.1
        a2 = ((i + 0.7) / num_dashes) * 2 * math.pi - t * 1.1
        p1 = (int(cx + r_mid * math.cos(a1)), int(cy + r_mid * math.sin(a1)))
        p2 = (int(cx + r_mid * math.cos(a2)), int(cy + r_mid * math.sin(a2)))
        cv2.line(out, p1, p2, col_s, 1, cv2.LINE_AA)

    # --- Inner solid ring ---
    cv2.circle(out, (cx, cy), r_in, col_p, 1, cv2.LINE_AA)

    # --- 24 radial ticks on outer ring ---
    for i in range(24):
        ang  = i * (2 * math.pi / 24) + t * 0.5
        long = i % 6 == 0
        ir   = r_out - (10 if long else 5)
        x1   = int(cx + ir      * math.cos(ang))
        y1   = int(cy + ir      * math.sin(ang))
        x2   = int(cx + r_out   * math.cos(ang))
        y2   = int(cy + r_out   * math.sin(ang))
        cv2.line(out, (x1, y1), (x2, y2), col_p if long else col_s, 1 if long else 1, cv2.LINE_AA)

    # --- Crosshair with centre gap ---
    gap  = 10
    clen = r_in - gap
    for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        cv2.line(out,
                 (cx + dx * gap,         cy + dy * gap),
                 (cx + dx * (gap + clen), cy + dy * (gap + clen)),
                 col_p, 1, cv2.LINE_AA)

    # --- Centre dot ---
    cv2.circle(out, (cx, cy), 3, col_a, -1, cv2.LINE_AA)


def _draw_side_gauges(out: np.ndarray, P: dict, h: int, w: int, t: float) -> None:
    """
    Segmented vertical bar gauges on left and right sides.
    Simulates a threat/distance meter — fill level pulses with time.
    """
    col_p  = P["primary"]
    col_s  = P["secondary"]
    segs   = 14
    seg_h  = int(h * 0.022)
    seg_w  = 6
    bar_h  = segs * (seg_h + 2)
    bar_y0 = (h - bar_h) // 2
    xs     = [18, w - 18 - seg_w]          # left and right x positions

    fill = int(segs * (0.55 + 0.3 * math.sin(t * 0.9)))
    fill = max(1, min(fill, segs))

    for bx in xs:
        for i in range(segs):
            fy  = bar_y0 + (segs - 1 - i) * (seg_h + 2)
            on  = i < fill
            col = col_p if on else (col_p[0] // 6, col_p[1] // 6, col_p[2] // 6)
            cv2.rectangle(out, (bx, fy), (bx + seg_w, fy + seg_h), col, -1)
        # label
        cv2.putText(out, "DST", (bx - 1, bar_y0 + bar_h + 14),
                    FONT, FONT_SCALE_SM * 0.7, col_s, FONT_THICKNESS, cv2.LINE_AA)


def _draw_target_box_at(out: np.ndarray, P: dict, bx: int, by: int, bw: int, bh: int, label: str, threat: str, t: float) -> None:
    """Helper to draw a single target bracket at specific coordinates."""
    col_a = P["accent"]
    col_s = P["secondary"]
    blink = int(t * 3) % 2 == 0
    col   = col_a if threat == "HIGH" else col_s
    draw  = (threat == "HIGH" and blink) or threat != "HIGH"
    if not draw:
        return

    arm = int(min(bw, bh) * 0.30)
    for sx2, sy2 in [(-1, -1), (1, -1), (-1, 1), (1, 1)]:
        px = bx + bw if sx2 == 1 else bx
        py = by + bh if sy2 == 1 else by
        cv2.line(out, (px + sx2 * arm, py), (px, py),             col, 1, cv2.LINE_AA)
        cv2.line(out, (px, py),             (px, py + sy2 * arm), col, 1, cv2.LINE_AA)
    cv2.putText(out, f"[{label}]", (bx, by - 5),
                FONT, FONT_SCALE_SM, col, FONT_THICKNESS, cv2.LINE_AA)
    cv2.putText(out, threat, (bx + bw - 30, by - 5),
                FONT, FONT_SCALE_SM, col, FONT_THICKNESS, cv2.LINE_AA)


def _draw_target_boxes(out: np.ndarray, P: dict, cx: int, cy: int, h: int, w: int, t: float) -> None:
    """
    Two target acquisition brackets (corner-only L-shapes) with labels.
    Boxes blink when threat level is HIGH.
    """
    targets = [
        {"x": cx - int(w * 0.30), "y": cy - int(h * 0.20), "bw": int(w * 0.10), "bh": int(h * 0.15),
         "label": "PREY-A", "threat": "HIGH"},
        {"x": cx + int(w * 0.20), "y": cy + int(h * 0.05), "bw": int(w * 0.09), "bh": int(h * 0.13),
         "label": "PREY-B", "threat": "LOW"},
    ]

    for tg in targets:
        _draw_target_box_at(out, P, tg["x"], tg["y"], tg["bw"], tg["bh"], tg["label"], tg["threat"], t)


def _draw_ecg_wave(out: np.ndarray, col: tuple, h: int, w: int, t: float) -> None:
    """Animated ECG polyline along the bottom of the frame."""
    pad   = 40
    x0, x1 = pad, w - pad
    y0    = h - int(h * 0.10)
    pts   = []
    # Use deterministic temporal seed for jitter stability
    rng_state = random.Random(int(t * 30))

    for x in range(x0, x1, 5):
        phase = ((x * 0.036) - t * 8.0) % (2 * math.pi)
        if 0.40 < phase < 0.68:
            dy = -int(h * 0.055) * math.sin((phase - 0.40) / 0.28 * math.pi)
        elif 0.68 < phase < 0.85:
            dy =  int(h * 0.018) * math.sin((phase - 0.68) / 0.17 * math.pi)
        elif 1.20 < phase < 1.65:
            dy = -int(h * 0.012) * math.sin((phase - 1.20) / 0.45 * math.pi)
        else:
            dy = 0
        # Simple pseudo-random jitter instead of heavy rng.normal
        dy += (rng_state.random() - 0.5) * 1.2
        pts.append([x, int(y0 + dy)])

    pts_arr = np.array(pts, dtype=np.int32)
    cv2.polylines(out, [pts_arr], False, col, 1, cv2.LINE_AA)

    bpm = int(82 + 5 * math.sin(t / 2.8))
    cv2.putText(out, f"PULSE {bpm} BPM", (x0, y0 - 10),
                FONT, FONT_SCALE_SM, col, FONT_THICKNESS, cv2.LINE_AA)


def _draw_signal_bar(out: np.ndarray, col: tuple, x: int, y: int, t: float) -> None:
    """Five-column animated signal-strength bars (like a phone indicator)."""
    sig = int(3 + 2 * abs(math.sin(t / 4.0)))   # 3–5 bars
    for i in range(5):
        bh = 4 + i * 3
        bx = x + i * 7
        by = y - bh
        c  = col if i < sig else (col[0] // 4, col[1] // 4, col[2] // 4)
        cv2.rectangle(out, (bx, by), (bx + 5, y), c, -1)


def _draw_telemetry(out: np.ndarray, P: dict, h: int, w: int, t: float, mode: str) -> None:
    """Left hex registers, right system flags, signal bar."""
    pad   = 20
    col_p = P["primary"]
    col_s = P["secondary"]
    pan_y = int(h * 0.18)
    lh    = int(h * 0.036)

    # --- Left panel: scrolling hex registers ---
    for i in range(6):
        v = ((int(t * 4) * 98761 + i * 13337) & 0xFFFF)
        cv2.putText(out, f"REG_{i:02d}: 0x{v:04X}",
                    (pad + 8, pan_y + i * lh),
                    FONT, FONT_SCALE_SM, col_p, FONT_THICKNESS, cv2.LINE_AA)

    # --- Right panel: status flags ---
    flags = [
        "SYS:    ONLINE",
        f"MODE:   {mode[:10]}",
        "OPTICS: ACTIVE",
        "SHIELD: 88 PCT",
        "COMM:   SECURE",
        "EMI:    99.4 PCT",
    ]
    rp_x = w - pad - 160
    for i, fl in enumerate(flags):
        cv2.putText(out, fl, (rp_x, pan_y + i * lh),
                    FONT, FONT_SCALE_SM, col_s, FONT_THICKNESS, cv2.LINE_AA)

    # Signal bar next to flags header
    _draw_signal_bar(out, col_p, rp_x + 130, pan_y - 6, t)


def _draw_header(out: np.ndarray, P: dict, h: int, w: int, t: float, mode: str) -> None:
    """Top bar: title left, mode label centre, power right."""
    pad   = 20
    ty    = int(h * 0.055)
    col_p = P["primary"]
    col_s = P["secondary"]
    col_a = P["accent"]

    # Separator line
    cv2.line(out, (pad + 8, ty + 7), (w - pad - 8, ty + 7), col_p, 1, cv2.LINE_AA)

    cv2.putText(out, f"{APP_NAME} v{APP_VERSION}",
                (pad + 12, ty),
                FONT, FONT_SCALE_MD, col_p, FONT_THICKNESS, cv2.LINE_AA)

    # Mode badge (centre) — blinks slowly
    badge = f"[ {mode} ]"
    bw, _ = cv2.getTextSize(badge, FONT, FONT_SCALE_MD, FONT_THICKNESS)
    bx    = w // 2 - bw[0] // 2
    alpha = 0.6 + 0.4 * abs(math.sin(t * 1.5))
    badge_col = tuple(int(c * alpha) for c in col_a)
    cv2.putText(out, badge, (bx, ty), FONT, FONT_SCALE_MD, badge_col, FONT_THICKNESS_BOLD, cv2.LINE_AA)

    pwr = int(90 + 3 * math.cos(t / 15.0))
    bar = "|" * (pwr // 10) + "." * (10 - pwr // 10)
    cv2.putText(out, f"PWR [{bar}] {pwr}%",
                (w - pad - 200, ty),
                FONT, FONT_SCALE_MD, col_s, FONT_THICKNESS, cv2.LINE_AA)


def _draw_motion_radar(out: np.ndarray, P: dict, h: int, w: int, t: float, targets: dict = None) -> None:
    """
    Cinematic Motion Tracker (bottom-right corner).
    Draws a radar sweep and 'pings' detected motion targets.
    """
    col_p = P["primary"]
    col_a = P["accent"]

    # Radar dimensions
    rad_size = int(min(h, w) * 0.18)
    pad = 30
    rx, ry = w - rad_size - pad, h - rad_size - pad
    cx, cy = rx + rad_size // 2, ry + rad_size // 2

    # Draw radar circle + grid
    cv2.circle(out, (cx, cy), rad_size // 2, col_p, 1, cv2.LINE_AA)
    cv2.circle(out, (cx, cy), rad_size // 4, col_p, 1, cv2.LINE_AA)
    cv2.line(out, (cx - rad_size // 2, cy), (cx + rad_size // 2, cy), col_p, 1)
    cv2.line(out, (cx, cy - rad_size // 2), (cx, cy + rad_size // 2), col_p, 1)

    # Radar sweep line
    sweep_ang = (t * 2.5) % (2 * math.pi)
    sx = int(cx + (rad_size // 2) * math.cos(sweep_ang))
    sy = int(cy + (rad_size // 2) * math.sin(sweep_ang))
    cv2.line(out, (cx, cy), (sx, sy), col_p, 2, cv2.LINE_AA)

    # Motion Pings
    motion_pts = []
    if isinstance(targets, dict):
        motion_pts = targets.get("motion", [])

    for pt in motion_pts:
        # Map frame coords (w, h) to radar polar coords
        mx, my = pt
        dx = (mx - w // 2) / (w // 2) # -1 to 1
        dy = (my - h // 2) / (h // 2) # -1 to 1

        px = int(cx + dx * (rad_size // 2.2))
        py = int(cy + dy * (rad_size // 2.2))

        # Only draw if within radar circle
        if math.sqrt((px - cx)**2 + (py - cy)**2) < rad_size // 2:
            cv2.circle(out, (px, py), 3, col_a, -1, cv2.LINE_AA)

    cv2.putText(out, "MOTION TRACKER", (rx, ry - 10),
                FONT, FONT_SCALE_SM, col_p, FONT_THICKNESS, cv2.LINE_AA)


def _draw_footer(out: np.ndarray, P: dict, h: int, w: int, t: float) -> None:
    """Bottom bar: GPS coords left, prey count right."""
    pad   = 20
    ty    = h - int(h * 0.035)
    col_p = P["primary"]
    col_s = P["secondary"]

    cv2.line(out, (pad + 8, ty - 7), (w - pad - 8, ty - 7), col_p, 1, cv2.LINE_AA)

    lat = 47.6062 + 2e-4 * math.sin(t / 4.0)
    lon = -122.3321 + 2e-4 * math.cos(t / 4.0)
    cv2.putText(out, f"LAT {lat:.5f}N  LON {lon:.5f}W  ALT 847m",
                (pad + 12, ty),
                FONT, FONT_SCALE_SM, col_p, FONT_THICKNESS, cv2.LINE_AA)

    cv2.putText(out, "PREY: 0  RANGE: ---  THREAT: LOW",
                (w - pad - 270, ty),
                FONT, FONT_SCALE_SM, col_s, FONT_THICKNESS, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Main HUD drawing function
# ---------------------------------------------------------------------------
def draw_hud(
    frame:          np.ndarray,
    mode:           str,
    error_state:    bool = False,
    routing_active: bool = False,
    targets:        dict = None
) -> np.ndarray:
    """
    Composite all Yautja HUD elements onto *frame*.

    Layers (bottom → top):
      1.  Hex mesh background
      2.  Side distance/threat gauges
      3.  Corner brackets with tick marks
      4.  Multi-ring rotating reticle
      5.  Target acquisition boxes   (TARGET_HUD / TACTICAL_ZOOM only)
      6.  ECG threat-pulse wave
      7.  Telemetry panels (hex registers + status flags + signal bar)
      8.  Header bar (title / mode badge / power)
      9.  Footer bar (GPS / prey count)
      10. Routing activity banner
      11. Error overlay
    """
    out         = frame.copy()
    h, w        = out.shape[:2]
    cx, cy      = w // 2, h // 2
    P           = _palette(mode)
    col         = _hud_color(mode, error_state)
    t           = time.time()

    # Determine if HUD should be simplified for low resolution
    # Threshold: 640x480 or smaller
    is_low_res = (w < 640 or h < 480)

    # ── 1. Hex mesh ─────────────────────────────────────────────────────────
    if not is_low_res:
        _draw_hex_mesh(out, mode if not error_state else "_default", alpha=0.06)

    # (No scanlines — preserves full image quality)

    # ── 2. Side gauges ──────────────────────────────────────────────────────
    if not is_low_res:
        _draw_side_gauges(out, P, h, w, t)

    # ── 3. Corner brackets ──────────────────────────────────────────────────
    _draw_corner_brackets(out, col, h, w)

    # ── 4. Rotating reticle ─────────────────────────────────────────────────
    if mode in ("TARGET_HUD", "AUTO_TARGET"):
        ret_x, ret_y = cx, cy
        if mode == "AUTO_TARGET" and targets and targets.get("faces"):
            # Track the first/largest face
            f = targets["faces"][0]
            ret_x = f[0] + f[2] // 2
            ret_y = f[1] + f[3] // 2

        _draw_reticle(out, P, ret_x, ret_y, t)

    # ── 5. Target boxes (mode-conditional) ──────────────────────────────────
    if mode in ("TARGET_HUD", "TACTICAL_ZOOM"):
        _draw_target_boxes(out, P, cx, cy, h, w, t)

    if mode == "AUTO_TARGET" and targets:
        for (fx, fy, fw, fh) in targets.get("faces", []):
            _draw_target_box_at(out, P, fx, fy, fw, fh, "BIO-SIGN", "HIGH", t)
        for (ex, ey, ew, eh) in targets.get("eyes", []):
            cv2.circle(out, (ex + ew // 2, ey + eh // 2), 4, P["accent"], 1, cv2.LINE_AA)

    # ── 6. ECG wave ─────────────────────────────────────────────────────────
    if not is_low_res:
        _draw_ecg_wave(out, col, h, w, t)

    # ── 7. Telemetry panels ──────────────────────────────────────────────────
    if not is_low_res:
        _draw_telemetry(out, P, h, w, t, mode)

    # ── 8. Header ────────────────────────────────────────────────────────────
    if not is_low_res:
        _draw_header(out, P, h, w, t, mode)

    # ── 9. Footer ───────────────────────────────────────────────────────────
    if not is_low_res:
        _draw_footer(out, P, h, w, t)
        _draw_motion_radar(out, P, h, w, t, targets=targets)

    # ── 10. Routing banner ───────────────────────────────────────────────────
    if routing_active and int(t * 3) % 2 == 0:
        bw2, bh2 = 400, 30
        bx1 = cx - bw2 // 2
        by1 = int(h * 0.08)
        cv2.rectangle(out, (bx1, by1), (bx1 + bw2, by1 + bh2), (0, 0, 0), -1)
        cv2.rectangle(out, (bx1, by1), (bx1 + bw2, by1 + bh2), col, 1)
        cv2.putText(out, "SYS: TELEMETRY INPUT ACTIVE — CHECK CONSOLE",
                    (bx1 + 10, by1 + 20),
                    FONT, FONT_SCALE_SM, col, FONT_THICKNESS, cv2.LINE_AA)

    # ── 11. Error overlay ────────────────────────────────────────────────────
    if error_state:
        ew, eh = 440, 76
        ex1 = cx - ew // 2
        ey1 = cy - eh // 2

        bg = out.copy()
        cv2.rectangle(bg, (ex1, ey1), (ex1 + ew, ey1 + eh), (0, 0, 50), -1)
        cv2.addWeighted(bg, 0.65, out, 0.35, 0, dst=out)

        cv2.rectangle(out, (ex1, ey1), (ex1 + ew, ey1 + eh), COLOR_RED, 2)
        cv2.putText(out, "BIO-MASK ERROR: TELEMETRY CORRUPTED",
                    (ex1 + 18, ey1 + 30),
                    FONT, FONT_SCALE_LG, COLOR_RED, FONT_THICKNESS_BOLD, cv2.LINE_AA)

        if int(t * 2.5) % 2 == 0:
            cv2.putText(out, "--- RECOVERY PROTOCOL ACTIVE ---",
                        (ex1 + 80, ey1 + 58),
                        FONT, FONT_SCALE_SM, COLOR_RED, FONT_THICKNESS, cv2.LINE_AA)

    return out