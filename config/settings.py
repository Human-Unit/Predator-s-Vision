"""
config/settings.py
==================
Central configuration for the Yautja-Vision Bio-Mask HUD.

All environment variables, API credentials, HUD color constants, and
default operational parameters are defined here. Import this module
anywhere in the codebase instead of hardcoding values.
"""

import os
from dotenv import load_dotenv

# Load .env file if it exists
load_dotenv()

# ---------------------------------------------------------------------------
# API / LLM Configuration
# Load from environment variables first; fall back to developer defaults.
# ---------------------------------------------------------------------------
API_KEY  = os.getenv("OPENAI_API_KEY", "YOUR_API_KEY_HERE")
BASE_URL = os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:1234/v1")  # LM Studio / Ollama default
MODEL    = os.getenv("OPENAI_MODEL",    "qwen/qwen2.5-vl-7b")

# ---------------------------------------------------------------------------
# Supported Visor Modes
# Must match the action identifiers used in the LLM system prompt exactly.
# ---------------------------------------------------------------------------
SUPPORTED_MODES = [
    "THERMAL_VISION",
    "TACTICAL_ZOOM",
    "TARGET_HUD",
    "AUTO_TARGET",
    "CLOAK_BLUR",
    "SPECTRUM_SHIFT",
    "NIGHT_VISION",
    "NORMAL_VISION",
]

# ---------------------------------------------------------------------------
# HUD Color Palette  (BGR format — OpenCV uses Blue, Green, Red channel order)
# ---------------------------------------------------------------------------
COLOR_GREEN  = (0, 255,   0)    # Primary neon HUD green
COLOR_RED    = (0,   0, 255)    # Alert / Target-lock red
COLOR_CYAN   = (255, 255,   0)  # Cloaking / tactical accent
COLOR_AMBER  = (0, 190, 255)    # Warm amber for thermal overlays
COLOR_WHITE  = (255, 255, 255)  # High-contrast labels
COLOR_BLACK  = (0,   0,   0)    # Solid background fills

# ---------------------------------------------------------------------------
# HUD Typography
# ---------------------------------------------------------------------------
FONT            = 0   # cv2.FONT_HERSHEY_SIMPLEX (imported in usage modules)
FONT_SCALE_SM   = 0.35
FONT_SCALE_MD   = 0.45
FONT_SCALE_LG   = 0.60
FONT_THICKNESS  = 1
FONT_THICKNESS_BOLD = 2

# ---------------------------------------------------------------------------
# Visor Defaults
# ---------------------------------------------------------------------------
DEFAULT_ZOOM_SCALE       = 2.0    # Tactical zoom magnification factor
DEFAULT_BLUR_STRENGTH    = 25     # Cloak blur Gaussian kernel radius
DEFAULT_SPECTRUM_TYPE    = "invert"  # UV spectrum inversion method
DEFAULT_IMAGE_PATH       = "photo.jpg"

# ---------------------------------------------------------------------------
# Glitch Configuration
# ---------------------------------------------------------------------------
GLITCH_FRAMES_TRANSITION = 6     # Frames of glitch on a successful mode switch
GLITCH_FRAMES_ERROR      = 12    # Frames of glitch on a telemetry error

# ---------------------------------------------------------------------------
# Application Meta
# ---------------------------------------------------------------------------
APP_NAME    = "YAUTJA BIO-MASK HUD"
APP_VERSION = "4.1"
WINDOW_NAME = f"{APP_NAME} v{APP_VERSION}"
