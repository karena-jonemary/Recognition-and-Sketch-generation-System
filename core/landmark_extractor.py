"""
core/landmark_extractor.py
Masked-aware facial landmark extraction using dlib.

This module provides:
  - dlib face detection (HOG/CNN)
  - 68-point landmark extraction
  - visibility scoring per landmark group
  - mask/occlusion heuristics
  - upper-face prioritisation when lower-face landmarks are likely hidden
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

try:
    import dlib
except Exception:  # pragma: no cover - handled at runtime with clear error
    dlib = None

try:
    import face_recognition_models
except Exception:  # pragma: no cover - optional helper package
    face_recognition_models = None


Point = Tuple[int, int]
BBox = Tuple[int, int, int, int]  # (top, right, bottom, left)

UPPER_FACE_GROUPS = {
    "left_eyebrow",
    "right_eyebrow",
    "left_eye",
    "right_eye",
    "nose_bridge",
    "nose_tip",
}
LOWER_FACE_GROUPS = {"chin", "jawline", "top_lip", "bottom_lip"}


@dataclass
class LandmarkExtractorConfig:
    detector_model: str = "hog"                  # "hog" | "cnn"
    upsample_times: int = 1
    visibility_threshold: float = 0.24
    upper_face_min_visibility: float = 0.14
    masked_confidence_threshold: float = 0.50
    predictor_path: Optional[str] = None


class DlibLandmarkExtractor:
    """
    dlib-based facial landmark extractor with masked-face awareness.

    Public API:
        - detect_faces(image) -> list[(top, right, bottom, left)]
        - extract(image, face_boxes=None) -> list[dict]
    """

    def __init__(self, config: Optional[LandmarkExtractorConfig] = None):
        if dlib is None:
            raise ImportError(
                "dlib is required for core.landmark_extractor. "
                "Install dlib (or face_recognition with dlib support)."
            )

        self.config = config or LandmarkExtractorConfig()
        self._hog_detector = dlib.get_frontal_face_detector()
        self._cnn_detector = self._init_cnn_detector()
        self._predictor = self._init_predictor()

    # ------------------------------------------------------------------
    # Model init
    # ------------------------------------------------------------------
    def _init_predictor(self):
        predictor_path = self.config.predictor_path or _resolve_predictor_path()
        if not predictor_path:
            raise RuntimeError(
                "Unable to resolve 68-point predictor path. "
                "Provide predictor_path explicitly in LandmarkExtractorConfig."
            )
        return dlib.shape_predictor(predictor_path)

    def _init_cnn_detector(self):
        if self.config.detector_model != "cnn":
            return None
        cnn_path = _resolve_cnn_detector_path()
        if not cnn_path:
            return None
        try:
            return dlib.cnn_face_detection_model_v1(cnn_path)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def detect_faces(self, image: np.ndarray) -> List[BBox]:
        """
        Detect faces and return bounding boxes as (top, right, bottom, left).
        """
        if image is None or image.size == 0:
            return []

        rgb = _to_rgb(image)
        h, w = rgb.shape[:2]

        if self.config.detector_model == "cnn" and self._cnn_detector is not None:
            detections = self._cnn_detector(rgb, self.config.upsample_times)
            rects = [det.rect for det in detections]
        else:
            rects = self._hog_detector(rgb, self.config.upsample_times)

        boxes: List[BBox] = []
        for rect in rects:
            left = int(max(0, rect.left()))
            top = int(max(0, rect.top()))
            right = int(min(w, rect.right() + 1))
            bottom = int(min(h, rect.bottom() + 1))
            if right <= left or bottom <= top:
                continue
            boxes.append((top, right, bottom, left))

        boxes.sort(key=lambda b: (b[2] - b[0]) * (b[1] - b[3]), reverse=True)
        return boxes

    def extract(
        self,
        image: np.ndarray,
        face_boxes: Optional[Sequence[BBox]] = None
    ) -> List[dict]:
        """
        Extract 68-point landmarks and visibility metadata.

        Returns a list with one dict per face:
            {
              "bbox": (top, right, bottom, left),
              "all_landmarks_68": [(x, y), ... 68],
              "landmarks": {group_name: [(x, y), ...]},
              "visible_landmarks": {...},
              "missing_landmarks": [...],
              "visibility_scores": {group_name: float},
              "is_masked": bool,
              "mask_confidence": float,
            }
        """
        if image is None or image.size == 0:
            return []

        boxes = list(face_boxes) if face_boxes else self.detect_faces(image)
        if not boxes:
            return []

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        results: List[dict] = []

        for box in boxes:
            top, right, bottom, left = _clamp_box(box, gray.shape[0], gray.shape[1])
            if right <= left or bottom <= top:
                continue

            rect = dlib.rectangle(left=left, top=top, right=right - 1, bottom=bottom - 1)
            shape = self._predictor(gray, rect)
            points68 = [(int(shape.part(i).x), int(shape.part(i).y)) for i in range(68)]
            grouped = _group_landmarks(points68)

            visibility_scores = self._compute_visibility_scores(gray, grouped, (top, right, bottom, left))
            is_masked, mask_confidence = self._estimate_masked_face(
                gray=gray,
                grouped=grouped,
                visibility_scores=visibility_scores,
                bbox=(top, right, bottom, left),
            )
            visible_landmarks, missing_groups = self._filter_visible_landmarks(
                grouped=grouped,
                visibility_scores=visibility_scores,
                is_masked=is_masked,
            )

            results.append({
                "bbox": (top, right, bottom, left),
                "all_landmarks_68": points68,
                "landmarks": grouped,
                "visible_landmarks": visible_landmarks,
                "missing_landmarks": missing_groups,
                "visibility_scores": visibility_scores,
                "is_masked": is_masked,
                "mask_confidence": mask_confidence,
            })

        return results

    # ------------------------------------------------------------------
    # Internal scoring
    # ------------------------------------------------------------------
    def _compute_visibility_scores(
        self,
        gray: np.ndarray,
        grouped: Dict[str, List[Point]],
        bbox: BBox
    ) -> Dict[str, float]:
        top, right, bottom, left = bbox
        visibility: Dict[str, float] = {}

        for group, pts in grouped.items():
            if len(pts) < 2:
                visibility[group] = 0.0
                continue

            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            x1 = max(left, min(xs) - 2)
            y1 = max(top, min(ys) - 2)
            x2 = min(right, max(xs) + 3)
            y2 = min(bottom, max(ys) + 3)

            if x2 <= x1 or y2 <= y1:
                visibility[group] = 0.0
                continue

            patch = gray[y1:y2, x1:x2]
            if patch.size == 0:
                visibility[group] = 0.0
                continue

            std_score = min(float(np.std(patch)) / 45.0, 1.0)
            edges = cv2.Canny(patch, 40, 120)
            edge_density = float(np.mean(edges > 0))
            edge_score = min(edge_density / 0.08, 1.0)

            # Promote structurally stable regions slightly
            size_score = min(len(pts) / 10.0, 1.0)
            score = (0.45 * std_score) + (0.45 * edge_score) + (0.10 * size_score)
            visibility[group] = float(max(0.0, min(score, 1.0)))

        return visibility

    def _estimate_masked_face(
        self,
        gray: np.ndarray,
        grouped: Dict[str, List[Point]],
        visibility_scores: Dict[str, float],
        bbox: BBox,
    ) -> Tuple[bool, float]:
        upper_scores = [visibility_scores.get(g, 0.0) for g in UPPER_FACE_GROUPS if g in grouped]
        lower_scores = [visibility_scores.get(g, 0.0) for g in LOWER_FACE_GROUPS if g in grouped]

        upper_mean = float(np.mean(upper_scores)) if upper_scores else 0.0
        lower_mean = float(np.mean(lower_scores)) if lower_scores else 0.0

        top, right, bottom, left = bbox
        mid_y = int((top + bottom) * 0.5)
        lower_roi = gray[mid_y:bottom, left:right]

        if lower_roi.size > 0:
            std_score = min(float(np.std(lower_roi)) / 40.0, 1.0)
            edge_density = float(np.mean(cv2.Canny(lower_roi, 40, 120) > 0))
            edge_score = min(edge_density / 0.06, 1.0)
            lower_texture_quality = 0.5 * std_score + 0.5 * edge_score
        else:
            lower_texture_quality = 0.0

        mask_confidence = 0.0
        if upper_mean > 0.35 and lower_mean < 0.22:
            mask_confidence += 0.45
        if upper_mean > (lower_mean + 0.18):
            mask_confidence += 0.30
        if upper_mean > 0.30 and lower_texture_quality < 0.22:
            mask_confidence += 0.25

        mask_confidence = float(max(0.0, min(mask_confidence, 1.0)))
        
        # If lips, jawline landmarks are clearly visible -> is_masked = False
        # If they are missing/low confidence -> is_masked = True
        if lower_mean > 0.25:
            is_masked = False
            mask_confidence = min(mask_confidence, 0.4)
        else:
            is_masked = True
            mask_confidence = max(mask_confidence, 0.6)
            
        print("Mask status:", is_masked)
        return is_masked, mask_confidence

    def _filter_visible_landmarks(
        self,
        grouped: Dict[str, List[Point]],
        visibility_scores: Dict[str, float],
        is_masked: bool,
    ) -> Tuple[Dict[str, List[Point]], List[str]]:
        visible: Dict[str, List[Point]] = {}
        missing = set()

        for group, pts in grouped.items():
            if len(pts) < 2:
                missing.add(group)
                continue

            group_score = visibility_scores.get(group, 0.0)

            if is_masked and group in LOWER_FACE_GROUPS:
                missing.add(group)
                continue

            if is_masked and group in UPPER_FACE_GROUPS:
                if group_score >= self.config.upper_face_min_visibility:
                    visible[group] = pts
                else:
                    missing.add(group)
                continue

            if group_score >= self.config.visibility_threshold:
                visible[group] = pts
            else:
                missing.add(group)

        # Keep jawline alias consistent with chin.
        if "chin" in visible and "jawline" not in visible:
            visible["jawline"] = visible["chin"]
        if "chin" in missing:
            missing.add("jawline")

        return visible, sorted(missing)


# ----------------------------------------------------------------------
# Convenience API
# ----------------------------------------------------------------------
_EXTRACTOR_CACHE: Dict[Tuple[str, str], DlibLandmarkExtractor] = {}


def get_landmark_extractor(
    model: str = "hog",
    predictor_path: Optional[str] = None,
    **config_overrides
) -> DlibLandmarkExtractor:
    """
    Get a cached DlibLandmarkExtractor instance.
    """
    cache_key = (model, predictor_path or "")
    if cache_key in _EXTRACTOR_CACHE:
        return _EXTRACTOR_CACHE[cache_key]

    cfg = LandmarkExtractorConfig(
        detector_model=model,
        predictor_path=predictor_path,
        **config_overrides,
    )
    extractor = DlibLandmarkExtractor(cfg)
    _EXTRACTOR_CACHE[cache_key] = extractor
    return extractor


def extract_landmarks(
    image: np.ndarray,
    face_boxes: Optional[Sequence[BBox]] = None,
    model: str = "hog",
    predictor_path: Optional[str] = None,
    **config_overrides
) -> List[dict]:
    """
    Convenience wrapper around DlibLandmarkExtractor.extract.
    """
    extractor = get_landmark_extractor(
        model=model,
        predictor_path=predictor_path,
        **config_overrides,
    )
    return extractor.extract(image, face_boxes=face_boxes)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _to_rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _clamp_box(box: BBox, h: int, w: int) -> BBox:
    top, right, bottom, left = box
    top = max(0, min(h, int(top)))
    right = max(0, min(w, int(right)))
    bottom = max(0, min(h, int(bottom)))
    left = max(0, min(w, int(left)))
    return top, right, bottom, left


def _group_landmarks(points68: Sequence[Point]) -> Dict[str, List[Point]]:
    """
    Build face_recognition-compatible group dict from 68 dlib points.
    """
    if len(points68) < 68:
        return {}

    points = list(points68)
    grouped = {
        "chin": points[0:17],
        "left_eyebrow": points[17:22],
        "right_eyebrow": points[22:27],
        "nose_bridge": points[27:31],
        "nose_tip": points[31:36],
        "left_eye": points[36:42],
        "right_eye": points[42:48],
        "top_lip": points[48:55] + [points[64], points[63], points[62], points[61], points[60]],
        "bottom_lip": points[54:60] + [points[48], points[60], points[67], points[66], points[65], points[64]],
    }
    grouped["jawline"] = grouped["chin"]
    return grouped


def _resolve_predictor_path() -> Optional[str]:
    if face_recognition_models is None:
        return None

    fn = getattr(face_recognition_models, "pose_predictor_model_location", None)
    if fn is None:
        return None

    try:
        return fn()
    except Exception:
        return None


def _resolve_cnn_detector_path() -> Optional[str]:
    if face_recognition_models is None:
        return None

    fn = getattr(face_recognition_models, "cnn_face_detector_model_location", None)
    if fn is None:
        return None

    try:
        return fn()
    except Exception:
        return None
