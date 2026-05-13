"""
core/preprocessor.py
OpenCV image preprocessing pipeline:
  1. Resize to max 640px wide (keeps aspect ratio)
  2. Fast non-local means denoising
  3. CLAHE contrast enhancement on L channel (LAB colour space)
"""

import cv2
import numpy as np


def preprocess(image: np.ndarray, max_width: int = 640) -> np.ndarray:
    """
    Full preprocessing pipeline for a BGR image.
    Returns a preprocessed BGR image ready for face detection.
    """
    if image is None or image.size == 0:
        return image

    # 1. Resize if too wide
    h, w = image.shape[:2]
    if w > max_width:
        scale = max_width / w
        image = cv2.resize(image, (max_width, int(h * scale)),
                           interpolation=cv2.INTER_AREA)

    # 2. Denoise (mild – keeps facial detail)
    denoised = cv2.fastNlMeansDenoisingColored(image, None,
                                               h=6, hColor=6,
                                               templateWindowSize=7,
                                               searchWindowSize=21)

    # 3. CLAHE contrast enhancement
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l_ch)
    enhanced = cv2.cvtColor(cv2.merge((l_enhanced, a_ch, b_ch)),
                            cv2.COLOR_LAB2BGR)

    return enhanced
