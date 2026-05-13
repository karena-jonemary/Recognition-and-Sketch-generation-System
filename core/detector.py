"""
core/detector.py
Face detection using dlib's HOG or CNN model via the face_recognition library.
  - HOG  → fast, CPU-friendly, good for well-lit frontal faces
  - CNN  → slower, more accurate, handles varied angles/lighting
"""

import numpy as np
import face_recognition


def detect_faces(image: np.ndarray, model: str = "hog") -> list:
    """
    Detect faces in a BGR image.

    Parameters
    ----------
    image : np.ndarray  BGR image
    model : str         "hog" (default) or "cnn"

    Returns
    -------
    list of (top, right, bottom, left) tuples
    """
    if image is None or image.size == 0:
        return []

    # face_recognition expects RGB
    rgb = cv2_bgr_to_rgb(image)
    locations = face_recognition.face_locations(rgb, model=model)
    return locations


def cv2_bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """Convert BGR numpy array to RGB."""
    return image[:, :, ::-1].copy()

