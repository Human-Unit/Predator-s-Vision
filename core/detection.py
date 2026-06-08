"""
core/detection.py
=================
Haar-cascade based face and eye detection for the Yautja Bio-Mask.
"""

import cv2
import numpy as np

# Load cascades once
_FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
_EYE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

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

                # Re-init tracker
                self.tracker = cv2.legacy.TrackerMOSSE_create()
                self.tracker.init(frame, tuple(self.last_rect))
            else:
                self.tracker = None
                self.last_rect = None

        # 2. Update tracker if active
        elif self.tracker is not None:
            success, rect = self.tracker.update(frame)
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
