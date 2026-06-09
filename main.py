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
import multiprocessing
import queue
import cv2
import numpy as np

from config.settings import (
    DEFAULT_IMAGE_PATH, WINDOW_NAME, APP_VERSION,
    GLITCH_FRAMES_TRANSITION, GLITCH_FRAMES_ERROR,
)
from core.llm_router   import YautjaRouter
from core.vision_engine import apply_mode
from core.detection    import TargetTracker, detect_gesture
from core.audio_engine import engine as audio
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
        self.recording        = False
        self.video_writer     = None

    # Convenience setters that acquire the lock
    def set_mode(self, mode: str, params: dict) -> None:
        with self._lock:
            if self.mode != mode:
                audio.play("MODE_CHANGE")
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

    if cmd.lower() == "record":
        with state._lock:
            state.recording = not state.recording
            if not state.recording and state.video_writer:
                state.video_writer.release()
                state.video_writer = None
                print("  [Bio-Mask] Recording stopped and saved.")
            elif state.recording:
                print("  [Bio-Mask] Recording started...")
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
# Multiprocessing Worker
# ---------------------------------------------------------------------------
def _vision_worker(input_q: multiprocessing.Queue, output_q: multiprocessing.Queue) -> None:
    """
    Sub-process that handles the heavy VisionEngine and HUD rendering.
    Bypasses the GIL to allow the main thread to focus on UI and I/O.
    """
    tracker = TargetTracker()
    bg_sub = cv2.createBackgroundSubtractorMOG2(history=50, varThreshold=25, detectShadows=False)
    frame_count = 0
    last_targets = None

    while True:
        try:
            # Get work: (frame, mode, params, error_state, routing_active)
            task = input_q.get(timeout=1.0)
            if task is None: break  # Sentinel

            frame, mode, params, error_state, routing_active, glitch_active = task

            # 1. Gesture Control (every 5 frames to save CPU)
            gesture = None
            if frame_count % 5 == 0:
                gesture = detect_gesture(frame)

            # 2. Detection / Tracking
            targets = {"faces": [], "eyes": [], "motion": []}
            force_detect = False

            if mode in ("AUTO_TARGET", "TARGET_HUD"):
                force_detect = (frame_count % 15 == 0)
                targets.update(tracker.update(frame, force_detect=force_detect))
                last_targets = targets

            frame_count += 1

            # 2. Motion Detection (for Radar)
            fg_mask = bg_sub.apply(frame)
            if frame_count % 2 == 0: # Downsample motion check
                contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    if cv2.contourArea(cnt) > 500:
                        mx, my, mw, mh = cv2.boundingRect(cnt)
                        targets["motion"].append([mx + mw//2, my + mh//2])

            # 3. Vision Filter
            processed = apply_mode(frame, mode, params)

            # 4. HUD Layer
            hud_frame = draw_hud(processed, mode, error_state, routing_active, targets=targets)

            # 5. Glitch Layer
            if glitch_active:
                hud_frame = apply_glitch(hud_frame)

            # 6. Lens Distortion (final post-process)
            final = apply_lens(hud_frame)

            # Send result back
            output_q.put((final, targets, force_detect, gesture))

        except queue.Empty:
            continue
        except Exception as e:
            print(f"  [Worker] Process error: {e}")




# ---------------------------------------------------------------------------
# Active mode — live webcam
# ---------------------------------------------------------------------------
def run_active_mode(router: YautjaRouter) -> None:
    state = VisorState()

    # Start Vision Worker Process
    input_q = multiprocessing.Queue(maxsize=2)
    output_q = multiprocessing.Queue(maxsize=2)
    worker = multiprocessing.Process(target=_vision_worker, args=(input_q, output_q), daemon=True)
    worker.start()

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

        # Offload rendering to worker process
        try:
            # Non-blocking put
            input_q.put_nowait((frame, state.mode, state.params, state.error_state, state.routing_active, state.consume_glitch_frame()))

            # Blocking get (wait for previous or current frame)
            rendered, targets, detected, gesture = output_q.get(timeout=0.1)

            # Handle gesture triggers in main thread
            if gesture == "OPEN_PALM":
                state.set_mode("CLOAK_BLUR", {"strength": 25})
            elif gesture == "PEACE":
                state.set_mode("THERMAL_VISION", {})
            elif gesture == "POINTING":
                state.set_mode("TARGET_HUD", {})

            # Handle sound in main thread
            if detected and targets and targets.get("faces"):
                if not state.last_targets or not state.last_targets.get("faces"):
                    audio.play("TARGET_LOCK")

            state.last_targets = targets

            # Handle Recording
            if state.recording:
                if state.video_writer is None:
                    h, w = rendered.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    state.video_writer = cv2.VideoWriter('combat_log.mp4', fourcc, 20.0, (w, h))
                state.video_writer.write(rendered)
                # Visual indicator for recording
                cv2.circle(rendered, (30, 30), 10, (0, 0, 255), -1, cv2.LINE_AA)

            cv2.imshow(WINDOW_NAME, rendered)
        except (queue.Full, queue.Empty):
            pass # Skip frame or wait

        key = cv2.waitKey(1) & 0xFF
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

    # Start Vision Worker Process
    input_q = multiprocessing.Queue(maxsize=2)
    output_q = multiprocessing.Queue(maxsize=2)
    worker = multiprocessing.Process(target=_vision_worker, args=(input_q, output_q), daemon=True)
    worker.start()

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
            input_q.put_nowait((base, state.mode, state.params, state.error_state, state.routing_active, state.consume_glitch_frame()))
            rendered, _, _, _ = output_q.get(timeout=0.1)
            cv2.imshow(WINDOW_NAME, rendered)
        except Exception as e:
            if not isinstance(e, (queue.Full, queue.Empty)):
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
