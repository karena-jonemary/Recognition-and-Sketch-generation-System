import os
import cv2
import numpy as np
import torch
from core.networks import UNetGenerator

class FaceCompleter:
    def __init__(self, weights_path=None, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.img_size = 256
        self.generator = UNetGenerator(in_channels=5, out_channels=3).to(self.device)
        self.generator.eval()
        
        if weights_path and os.path.exists(weights_path):
            self.generator.load_state_dict(torch.load(weights_path, map_location=self.device))
            print(f"Loaded face completion weights from {weights_path}")
        else:
            print("Warning: No pretrained weights found for Face Completer. Using untrained model.")

    def _generate_heatmap(self, landmarks, h, w):
        heatmap = np.zeros((h, w), dtype=np.float32)
        for (x, y) in landmarks:
            if 0 <= x < w and 0 <= y < h:
                sigma = 3
                size = 6
                x0, y0 = max(0, x - size), max(0, y - size)
                x1, y1 = min(w, x + size + 1), min(h, y + size + 1)
                for i in range(y0, y1):
                    for j in range(x0, x1):
                        heatmap[i, j] = max(heatmap[i, j], np.exp(-((j - x)**2 + (i - y)**2) / (2 * sigma**2)))
        return heatmap

    def _create_occlusion_mask(self, h, w, landmarks):
        """Estimate the occluded region based on visible upper landmarks."""
        mask = np.zeros((h, w), dtype=np.float32)
        
        # If we have nose landmarks, use them to estimate mask start
        # Assume missing lower face (y-coordinate)
        ys = [p[1] for p in landmarks]
        if ys:
            max_visible_y = max(ys)
            # Add a small buffer below the lowest visible feature
            mask_start_y = min(h - 10, max_visible_y + int(h * 0.05))
        else:
            mask_start_y = int(h * 0.5)
            
        # Draw ellipse for the lower face
        center = (w // 2, mask_start_y + 10)
        axes = (w // 2, h - mask_start_y + 20)
        cv2.ellipse(mask, center, axes, 0, 0, 180, 1.0, -1)
        
        mask[mask_start_y:, :] = np.where(mask[mask_start_y:, :] == 0, 1.0, mask[mask_start_y:, :])
        return mask

    def complete_face(self, bgr_img, landmarks_68):
        """
        Completes the lower half of a masked face.
        Args:
            bgr_img: The cropped face image (numpy array, BGR)
            landmarks_68: List of (x, y) tuples for visible landmarks.
        Returns:
            reconstructed_img: The complete face image (numpy array, BGR)
        """
        orig_h, orig_w = bgr_img.shape[:2]
        
        # 1. Resize image and scale landmarks to 256x256
        img_rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (self.img_size, self.img_size))
        
        scaled_landmarks = []
        for (x, y) in landmarks_68:
            sx = int(x * self.img_size / orig_w)
            sy = int(y * self.img_size / orig_h)
            scaled_landmarks.append((sx, sy))
            
        # 2. Generate Heatmap and Mask
        heatmap = self._generate_heatmap(scaled_landmarks, self.img_size, self.img_size)
        mask = self._create_occlusion_mask(self.img_size, self.img_size, scaled_landmarks)
        
        # 3. Prepare Tensors
        # Image to [-1, 1]
        img_tensor = (img_resized.astype(np.float32) / 127.5) - 1.0
        
        # Apply mask to image (set masked region to 0)
        for c in range(3):
            img_tensor[:, :, c] = img_tensor[:, :, c] * (1 - mask)
            
        img_tensor = torch.from_numpy(img_tensor).permute(2, 0, 1).unsqueeze(0).to(self.device)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).to(self.device)
        heatmap_tensor = torch.from_numpy(heatmap).unsqueeze(0).unsqueeze(0).to(self.device)
        
        # Combine inputs [1, 5, 256, 256]
        gen_input = torch.cat([img_tensor, mask_tensor, heatmap_tensor], dim=1)
        
        # 4. Inference
        with torch.no_grad():
            output_tensor = self.generator(gen_input)
            
        # 5. Post-process
        output_img = output_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        output_img = ((output_img + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
        
        # 6. Blend: use original image for unmasked regions, generated for masked
        # Smooth the mask for blending
        smooth_mask = cv2.GaussianBlur(mask, (15, 15), 0)
        smooth_mask = np.expand_dims(smooth_mask, axis=2)
        
        blended = img_resized * (1 - smooth_mask) + output_img * smooth_mask
        blended = blended.astype(np.uint8)
        
        # 7. Resize back to original
        final_bgr = cv2.cvtColor(blended, cv2.COLOR_RGB2BGR)
        final_bgr = cv2.resize(final_bgr, (orig_w, orig_h))
        
        return final_bgr

# Singleton access
_COMPLETER_INSTANCE = None

def get_face_completer(weights_path="checkpoints/generator_pretrained.pth"):
    global _COMPLETER_INSTANCE
    if _COMPLETER_INSTANCE is None:
        _COMPLETER_INSTANCE = FaceCompleter(weights_path=weights_path)
    return _COMPLETER_INSTANCE

def complete_face(bgr_img, landmarks_68):
    completer = get_face_completer()
    return completer.complete_face(bgr_img, landmarks_68)
