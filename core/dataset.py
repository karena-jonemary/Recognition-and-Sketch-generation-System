import os
import cv2
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
import random

class MaskedFaceDataset(Dataset):
    def __init__(self, image_dir, transform=None, img_size=256):
        """
        Args:
            image_dir: Path to original full faces (e.g. CelebA/FFHQ)
        """
        self.image_paths = glob.glob(os.path.join(image_dir, "*.jpg")) + \
                           glob.glob(os.path.join(image_dir, "*.png"))
        self.transform = transform
        self.img_size = img_size
        
        # dlib detector for generating landmarks on the fly (or load precomputed)
        try:
            import dlib
            self.detector = dlib.get_frontal_face_detector()
            # Default path, you should pass a valid path in production
            self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") 
        except Exception:
            self.detector = None

    def __len__(self):
        return len(self.image_paths)
        
    def _create_lower_face_mask(self, h, w):
        """Simulate a mask covering the lower face."""
        mask = np.zeros((h, w), dtype=np.float32)
        
        # Randomize mask height (between 40% and 60% of the lower face)
        mask_h_start = int(h * random.uniform(0.4, 0.6))
        
        # Ellipse to simulate realistic mask boundary
        center = (w // 2, mask_h_start + 20)
        axes = (w // 2 + 10, h - mask_h_start + 20)
        cv2.ellipse(mask, center, axes, 0, 0, 180, 1.0, -1)
        
        # Ensure bottom is covered
        mask[mask_h_start:, :] = np.where(mask[mask_h_start:, :] == 0, 1.0, mask[mask_h_start:, :])
        return mask

    def _generate_heatmap(self, landmarks, h, w):
        """Generate a single heatmap channel for all 68 landmarks."""
        heatmap = np.zeros((h, w), dtype=np.float32)
        for (x, y) in landmarks:
            if 0 <= x < w and 0 <= y < h:
                # Add a gaussian dot
                sigma = 3
                size = 6
                x0, y0 = max(0, x - size), max(0, y - size)
                x1, y1 = min(w, x + size + 1), min(h, y + size + 1)
                for i in range(y0, y1):
                    for j in range(x0, x1):
                        heatmap[i, j] = max(heatmap[i, j], np.exp(-((j - x)**2 + (i - y)**2) / (2 * sigma**2)))
        return heatmap

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = cv2.imread(img_path)
        if img is None:
            return self.__getitem__((idx + 1) % len(self))
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size))
        
        # Extract landmarks
        landmarks = []
        if self.detector is not None:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            rects = self.detector(gray, 1)
            if len(rects) > 0:
                shape = self.predictor(gray, rects[0])
                landmarks = [(shape.part(i).x, shape.part(i).y) for i in range(68)]
        
        # Fallback to empty landmarks if none detected
        if len(landmarks) == 0:
            return self.__getitem__((idx + 1) % len(self))
            
        heatmap = self._generate_heatmap(landmarks, self.img_size, self.img_size)
        mask = self._create_lower_face_mask(self.img_size, self.img_size)
        
        # Create masked image
        img_masked = img.copy().astype(np.float32)
        # Apply black mask
        for c in range(3):
            img_masked[:, :, c] = img_masked[:, :, c] * (1 - mask)
            
        # Normalize to [-1, 1]
        img_real = (img.astype(np.float32) / 127.5) - 1.0
        img_masked = (img_masked / 127.5) - 1.0
        
        # Convert to tensors
        img_real = torch.from_numpy(img_real).permute(2, 0, 1)
        img_masked = torch.from_numpy(img_masked).permute(2, 0, 1)
        mask = torch.from_numpy(mask).unsqueeze(0)
        heatmap = torch.from_numpy(heatmap).unsqueeze(0)
        
        # Concatenate inputs: RGB (3) + Mask (1) + Heatmap (1) = 5 channels
        generator_input = torch.cat([img_masked, mask, heatmap], dim=0)
        
        return {
            "input": generator_input,
            "target": img_real,
            "mask": mask,
            "masked_img": img_masked
        }
