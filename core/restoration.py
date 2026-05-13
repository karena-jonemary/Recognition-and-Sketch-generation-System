"""
core/restoration.py
Face RESTORATION module using GFPGAN.
"""

import os
import cv2
import numpy as np

try:
    import torch
    from gfpgan import GFPGANer
except ImportError:
    torch = None
    GFPGANer = None

_restorer = None

def init_restorer():
    """Initialise the GFPGANer model (loads weights, sets up network)."""
    global _restorer
    if GFPGANer is None:
        print("GFPGAN library not installed.")
        return

    # User requested to start with CPU
    device = torch.device('cpu')
    
    print("Initialising GFPGAN Restorer on CPU...")
    # upscale=1 means we return the face at the same scale to avoid blowing up memory boundaries
    try:
        _restorer = GFPGANer(
            model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth',
            upscale=1,
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=None,
            device=device
        )
        print("GFPGAN Restorer Initialised successfully.")
    except Exception as e:
        print(f"Failed to initialise GFPGAN: {e}")

def enhance_face(image: np.ndarray) -> np.ndarray:
    """
    Takes an input face crop (BGR), runs it through GFPGAN, and returns 
    the restored face crop.
    Falls back to original image if enhancement fails.
    """
    global _restorer
    if _restorer is None:
        init_restorer()
        if _restorer is None:
            return image

    if image is None or image.size == 0:
        return image

    try:
        # enhance method returns: cropped_faces, restored_faces, restored_img
        # since we pass just the face crop, restored_img is the restored view.
        _, _, restored_img = _restorer.enhance(
            image, 
            has_aligned=False, 
            only_center_face=True, 
            paste_back=True
        )
        if restored_img is not None:
            return restored_img
    except Exception as exc:
        print(f"GFPGAN Exception during enhancement: {exc}")
    
    return image
