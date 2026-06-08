"""
main.py
=======
Yautja-Vision Bio-Mask HUD — Application Entry Point

Dual-mode orchestration:
  Active Mode  — live webcam stream (cv2.VideoCapture(0))
  Passive Mode — static image file (cv2.imread)

Threading model:
  The main thread owns the OpenCV window and renders frames at ~30 FPS.
  When the user presses 'C', a daemon thread handles terminal input and
  a second daemon thread submits the LLM request — the video loop is never
  blocked by I/O or network latency.

Controls (focus the OpenCV window):
  C — enter a new natural-language visor command in the terminal
  Q — quit the application cleanly
"""

import sys
import threading
import cv2
import numpy as np

from config.settings import (
    DEFAULT_IMAGE_PATH, WINDOW_NAME, APP_VERSION,
    GLITCH_FRAMES_TRANSITION, GLITCH_FRAMES_ERROR,
)
from core.llm_router   import YautjaRouter
from core.vision_engine import apply_mode
from core.detection    import detect_targets
from ui.hud_overlays   import draw_hud, apply_lens, apply_glitch


# ---------------------------------------------------------------------------
# Shared visor state  (written by background threads, read by the render loop)
# ---------------------------------------------------------------------------
class VisorState:
    """Thread-safe container for the current visor operating state."""

    def __init__(self) -> None:
        self._lock            = threading.Lock()
        self.mode             = "NORMAL_VISION"
        self.params: dict     = {}
        self.error_state      = False
        self.routing_active   = False
        self.glitch_frames    = 0
        self.exit_requested   = False
        self.last_targets     = None
        self.detection_frame_count = 0

    # Convenience setters that acquire the lock
    def set_mode(self, mode: str, params: dict) -> None:
        with self._lock:
            self.mode          = mode
            self.params        = params
            self.error_state   = False
            self.glitch_frames = GLITCH_FRAMES_TRANSITION

    def set_error(self) -> None:
        with self._lock:
            self.mode          = "NORMAL_VISION"
            self.params        = {}
            self.error_state   = True
            self.glitch_frames = GLITCH_FRAMES_ERROR

    def consume_glitch_frame(self) -> bool:
        """Return True if a glitch frame should be rendered, decrement counter."""
        with self._lock:
            if self.glitch_frames > 0:
                self.glitch_frames -= 1
                return True
            return False


# ---------------------------------------------------------------------------
# Background thread: terminal input
# ---------------------------------------------------------------------------
def _prompt_and_route(state: VisorState, router: YautjaRouter) -> None:
    """
    Runs in a daemon thread.
    1. Read a line from stdin (blocks, but only this thread).
    2. If not empty, spawn another daemon thread to call the LLM router.
    3. Clear routing_active flag when finished.
    """
    try:
        sys.stdout.write("\n>> Enter bio-mask telemetry command: ")
        sys.stdout.flush()
        cmd = sys.stdin.readline().strip()
    except Exception:
        state.routing_active = False
        return

    if cmd.lower() in ("quit", "exit", "q"):
        state.exit_requested = True
        state.routing_active = False
        return

    if not cmd:
        state.routing_active = False
        return

    # Route the command in yet another daemon thread so stdin returns immediately
    def _route() -> None:
        try:
            payload = router.route_command(cmd)
            state.set_mode(payload["action"], payload["parameters"])
            print(f"  [Bio-Mask] Mode → {payload['action']}  params={payload['parameters']}")
        except Exception as exc:
            print(f"  [Bio-Mask] Telemetry route failed: {exc}")
            state.set_error()
        finally:
            state.routing_active = False

    threading.Thread(target=_route, daemon=True).start()


def trigger_command_input(state: VisorState, router: YautjaRouter) -> None:
    """Launch the input+routing daemon thread (called from the render loop)."""
    if state.routing_active:
        return
    state.routing_active = True
    threading.Thread(target=_prompt_and_route, args=(state, router), daemon=True).start()


# ---------------------------------------------------------------------------
# Frame render pipeline  (shared by both modes)
# ---------------------------------------------------------------------------
def _render(frame: np.ndarray, state: VisorState) -> np.ndarray:
    """Apply vision filter → HUD → optional glitch → lens distortion."""
    try:
        targets = None
        if state.mode == "AUTO_TARGET":
            # Optimization: Run detection every 3 frames to save CPU
            if state.detection_frame_count % 3 == 0:
                state.last_targets = detect_targets(frame)
            targets = state.last_targets
            state.detection_frame_count += 1

        processed = apply_mode(frame, state.mode, state.params)
        hud_frame = draw_hud(processed, state.mode, state.error_state, state.routing_active, targets=targets)
        if state.consume_glitch_frame():
            hud_frame = apply_glitch(hud_frame)
        return apply_lens(hud_frame)
    except Exception as e:
        print(f"  [Bio-Mask] Render Error: {e}")
        state.set_error()
        return frame


# ---------------------------------------------------------------------------
# Active mode — live webcam
# ---------------------------------------------------------------------------
def run_active_mode(router: YautjaRouter) -> None:
    state = VisorState()

    print("\n[SYS] Initialising camera…")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[warn] Webcam unavailable.")
        choice = input("Switch to Passive Mode? (y/n): ").strip().lower()
        if choice.startswith("y"):
            run_passive_mode(router)
        return

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    print(f"\n  [{WINDOW_NAME}] Live stream active.")
    print("  Focus the window → C = new command | Q = quit\n")

    while not state.exit_requested:
        ret, frame = cap.read()
        if not ret:
            print("[error] Lost webcam frame.")
            break

        # Remove mirror effect (standard for webcams in OpenCV)
        frame = cv2.flip(frame, 1)

        cv2.imshow(WINDOW_NAME, _render(frame, state))

        key = cv2.waitKey(30) & 0xFF
        if key in (ord("q"), ord("Q")):
            print("[bye] Shutting down telemetry feed.")
            break
        if key in (ord("c"), ord("C")):
            trigger_command_input(state, router)

    cap.release()
    cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Passive mode — static image
# ---------------------------------------------------------------------------
def run_passive_mode(router: YautjaRouter) -> None:
    state = VisorState()

    raw_path = input(f"Image path [default: {DEFAULT_IMAGE_PATH}]: ").strip()
    image_path = raw_path or DEFAULT_IMAGE_PATH

    base = cv2.imread(image_path)
    if base is None:
        print(f"[error] Cannot load image: {image_path!r}")
        sys.exit(1)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    print(f"\n  [{WINDOW_NAME}] Static visualiser active.")
    print("  Focus the window → C = new command | Q = quit\n")

    while not state.exit_requested:
        try:
            rendered = _render(base, state)
            cv2.imshow(WINDOW_NAME, rendered)
        except Exception as e:
            print(f"[error] Passive Mode display failed: {e}")
            break

        key = cv2.waitKey(30) & 0xFF
        if key in (ord("q"), ord("Q")):
            print("[bye] Closing visualiser.")
            break
        if key in (ord("c"), ord("C")):
            trigger_command_input(state, router)

    cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 60)
    print(f"   YAUTJA-VISION BIO-MASK TELEMETRY SYSTEM v{APP_VERSION}")
    print("=" * 60)

    router = YautjaRouter()

    print("\nSelect Visor Mode:")
    print("  [1]  Active Mode   — Webcam (real-time)")
    print("  [2]  Passive Mode  — Static image")

    try:
        choice = input("\nSelect [1/2]: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n[bye]")
        return

    if choice == "1":
        run_active_mode(router)
    elif choice == "2":
        run_passive_mode(router)
    else:
        print("[warn] Unrecognised choice — defaulting to Active Mode.")
        run_active_mode(router)


if __name__ == "__main__":
    main()
