"""
core/encoder.py
128-dimensional face encoding and 68-point landmark extraction
using dlib (via the face_recognition library).
"""

import numpy as np
import face_recognition
from core.detector import cv2_bgr_to_rgb


def encode_faces(image: np.ndarray, locations: list) -> list:
    """
    Generate 128D face encodings for detected face locations.

    Parameters
    ----------
    image     : BGR numpy array
    locations : list of (top, right, bottom, left) from detector

    Returns
    -------
    list of 128-element numpy arrays (one per face)
    """
    if not locations:
        return []
    rgb = cv2_bgr_to_rgb(image)
    return face_recognition.face_encodings(rgb, locations)


def get_landmarks(image: np.ndarray, locations: list = None) -> list:
    """
    Extract 68 facial landmarks for each detected face.

    Returns
    -------
    list of dicts, each with keys:
        chin, left_eyebrow, right_eyebrow, nose_bridge, nose_tip,
        left_eye, right_eye, top_lip, bottom_lip
    Each value is a list of (x, y) pixel coordinates.
    """
    rgb = cv2_bgr_to_rgb(image)
    return face_recognition.face_landmarks(rgb, locations)
