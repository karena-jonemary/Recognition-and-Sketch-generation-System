"""
core/sketcher.py
Generates a pencil-sketch representation of an unknown face by:
  1. Drawing the 68 dlib landmark contours (jaw, eyes, nose, mouth, eyebrows)
     onto a blank canvas using OpenCV polylines.
  2. Applying a classic pencil-sketch effect to the face crop.
  3. Blending the two layers for the final output.
"""

import cv2
import numpy as np


# Landmark groups and whether the polyline is closed
_GROUPS = [
    ("chin",           False),
    ("left_eyebrow",   False),
    ("right_eyebrow",  False),
    ("nose_bridge",    False),
    ("nose_tip",       False),
    ("left_eye",       True),
    ("right_eye",      True),
    ("top_lip",        True),
    ("bottom_lip",     True),
]


def landmarks_to_sketch(face_crop: np.ndarray,
                         landmarks: dict) -> np.ndarray:
    """
    Produce a sketch that combines landmark line-art with a pencil effect.

    Parameters
    ----------
    face_crop : np.ndarray
        BGR crop of the face (already offset to top-left = (0,0)).
    landmarks : dict
        Landmark dict whose coordinates are relative to *face_crop*
        (i.e. already adjusted by subtracting the face's (left, top)).

    Returns
    -------
    np.ndarray  BGR sketch image, same size as face_crop.
    """
    if face_crop is None or face_crop.size == 0:
        return face_crop

    h, w = face_crop.shape[:2]

    # --- Layer 1: pencil sketch of the original face ---
    pencil = _pencil_sketch(face_crop)

    # --- Layer 2: draw bold black landmark contour lines directly onto the sketch ---
    for group_name, closed in _GROUPS:
        pts = landmarks.get(group_name, [])
        if len(pts) < 2:
            continue
        arr = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(pencil, [arr], closed,
                      (0, 0, 0), 2, cv2.LINE_AA)

    return pencil


def generate_sketch(face_crop: np.ndarray) -> np.ndarray:
    """
    Pencil-sketch fallback when no landmarks are available.
    """
    return _pencil_sketch(face_crop)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _pencil_sketch(image: np.ndarray) -> np.ndarray:
    """Improved dodge-and-burn pencil-sketch effect."""
    # Smooth to remove noise/pores before sketching
    smoothed = cv2.medianBlur(image, 3)
    gray = cv2.cvtColor(smoothed, cv2.COLOR_BGR2GRAY)
    
    # Dodge and burn
    inverted = cv2.bitwise_not(gray)
    blurred = cv2.GaussianBlur(inverted, (21, 21), 0)
    inv_blur = cv2.bitwise_not(blurred)
    sketch_gray = cv2.divide(gray, inv_blur, scale=256.0)

    # Deepen the pencil strokes using gamma correction
    # A gamma > 1 darkens midtones significantly while keeping white as white
    gamma = 2.5
    lookUpTable = np.empty((1, 256), np.uint8)
    for i in range(256):
        lookUpTable[0, i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
    sketch_gray = cv2.LUT(sketch_gray, lookUpTable)

    return cv2.cvtColor(sketch_gray, cv2.COLOR_GRAY2BGR)
