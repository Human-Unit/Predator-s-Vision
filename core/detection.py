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

def detect_targets(frame: np.ndarray) -> dict:
    """
    Detect faces and eyes in the frame.
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
