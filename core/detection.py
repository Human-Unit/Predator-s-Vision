"""
core/detection.py
=================
Haar-cascade based face and eye detection for the Yautja Bio-Mask.
"""

import math
import cv2
import numpy as np
import mediapipe as mp
# Explicit import for compatibility with newer mediapipe versions
try:
    import mediapipe.python.solutions.hands as mp_hands
except ImportError:
    mp_hands = mp.solutions.hands

# Load cascades once
_FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
_EYE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Mediapipe Hands
_HANDS = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

def detect_gesture(frame: np.ndarray) -> str:
    """
    Detect hand gestures to trigger visor modes.
    Extended Mapping:
      - OPEN_PALM  -> All fingers up
      - FIST       -> All fingers down
      - PEACE      -> Index + Middle up
      - POINTING   -> Index up
      - ROCK_ON    -> Index + Pinky up
      - THUMBS_UP  -> Thumb up, others down
      - OK_SIGN    -> Thumb + Index tips touching
    """
    results = _HANDS.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if not results.multi_hand_landmarks:
        return None

    landmarks = results.multi_hand_landmarks[0].landmark

    # Finger states (True = extended)
    # Thumb: distance to pinky base
    thumb_open = math.sqrt((landmarks[4].x - landmarks[17].x)**2 + (landmarks[4].y - landmarks[17].y)**2) > 0.15

    # Tips: 8 (Index), 12 (Middle), 16 (Ring), 20 (Pinky)
    # PIPs: 6 (Index), 10 (Middle), 14 (Ring), 18 (Pinky)
    f_index  = landmarks[8].y < landmarks[6].y
    f_middle = landmarks[12].y < landmarks[10].y
    f_ring   = landmarks[16].y < landmarks[14].y
    f_pinky  = landmarks[20].y < landmarks[18].y

    # 1. OK SIGN (Special distance check)
    dist_thumb_index = math.sqrt((landmarks[4].x - landmarks[8].x)**2 + (landmarks[4].y - landmarks[8].y)**2)
    if dist_thumb_index < 0.05 and f_middle and f_ring and f_pinky:
        return "OK_SIGN"

    # 2. OPEN PALM
    if thumb_open and f_index and f_middle and f_ring and f_pinky:
        return "OPEN_PALM"

    # 3. FIST
    if not thumb_open and not f_index and not f_middle and not f_ring and not f_pinky:
        return "FIST"

    # 4. PEACE
    if f_index and f_middle and not f_ring and not f_pinky:
        return "PEACE"

    # 5. POINTING
    if f_index and not f_middle and not f_ring and not f_pinky:
        return "POINTING"

    # 6. ROCK ON
    if f_index and f_pinky and not f_middle and not f_ring:
        return "ROCK_ON"

    # 7. THUMBS UP
    if thumb_open and not f_index and not f_middle and not f_ring and not f_pinky:
        # Check if thumb is actually pointing up relative to wrist
        if landmarks[4].y < landmarks[3].y:
            return "THUMBS_UP"

    return None

class TargetTracker:
    """
    Hybrid detection/tracking system:
    - Uses Haar cascades for initial discovery and periodic correction.
    - Uses MOSSE tracker for high-speed tracking on intermediate frames.
    """
    def __init__(self) -> None:
        self.tracker = None
        self.last_rect = None

    def update(self, frame: np.ndarray, force_detect: bool = False) -> dict:
        """
        Update tracking state. Runs detection if tracker is lost or force_detect is True.
        """
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 1. Periodically force detection to prevent drift or find new targets
        if force_detect or self.tracker is None:
            faces = _FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            if len(faces) > 0:
                # Prioritize largest face
                faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
                target = faces[0]
                self.last_rect = [int(x) for x in target]

                # Re-init tracker with error handling
                try:
                    self.tracker = cv2.legacy.TrackerMOSSE_create()
                    self.tracker.init(frame, tuple(self.last_rect))
                except Exception as e:
                    print(f"  [Tracker] Initialization failed: {e}")
                    self.tracker = None
            else:
                self.tracker = None
                self.last_rect = None

        # 2. Update tracker if active
        elif self.tracker is not None:
            try:
                success, rect = self.tracker.update(frame)
            except Exception as e:
                print(f"  [Tracker] Update failed: {e}")
                success = False
            if success:
                self.last_rect = [int(x) for x in rect]
            else:
                self.tracker = None
                self.last_rect = None

        # 3. Handle Eye detection (only if a face is active)
        all_eyes = []
        if self.last_rect:
            x, y, fw, fh = self.last_rect
            # Safety clamp for tracker drift
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(w, x+fw), min(h, y+fh)

            if x2 > x1 and y2 > y1:
                roi_gray = gray[y1:y2, x1:x2]
                eyes = _EYE_CASCADE.detectMultiScale(roi_gray)
                for (ex, ey, ew, eh) in eyes:
                    all_eyes.append([int(x1 + ex), int(y1 + ey), int(ew), int(eh)])

        return {
            "faces": [self.last_rect] if self.last_rect else [],
            "eyes": all_eyes
        }

def detect_targets(frame: np.ndarray) -> dict:
    """
    Legacy stateless detection.
    Returns a dict with 'faces' (list of [x, y, w, h]) and 'eyes' (list of [x, y, w, h]).
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = _FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Sort faces by area (w * h) descending to prioritize closest target
    if len(faces) > 0:
        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)

    all_eyes = []
    # Convert faces to list for JSON/thread-safety if needed
    faces_list = []
    for (x, y, w, h) in faces:
        faces_list.append([int(x), int(y), int(w), int(h)])

        # Region of interest for eyes within the face
        roi_gray = gray[y:y+h, x:x+w]
        eyes = _EYE_CASCADE.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            all_eyes.append([int(x + ex), int(y + ey), int(ew), int(eh)])

    return {
        "faces": faces_list,
        "eyes": all_eyes
    }
