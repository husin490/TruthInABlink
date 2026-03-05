"""
Truth in a Blink — Robust Face Detection & Cropping
=====================================================
Unified face detection API with three-tier fallback:
    1. MediaPipe Face Detection (best — handles lighting/pose)
    2. Haar Cascade (fast fallback)
    3. Centre crop (last resort, flagged low-confidence)

Usage
-----
    from utils.face_crop import detect_face
    crop, bbox, confidence, method = detect_face(bgr_image)
"""

import logging
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ── MediaPipe (lazy import — optional dependency) ─────────────────────────────

_mp_face_detection = None
_mp_detector = None
_MP_AVAILABLE = False

try:
    import mediapipe as mp
    _MP_AVAILABLE = True
except ImportError:
    pass


def _get_mp_detector():
    """Lazy-initialise MediaPipe face detector (singleton)."""
    global _mp_face_detection, _mp_detector
    if _mp_detector is None and _MP_AVAILABLE:
        _mp_face_detection = mp.solutions.face_detection
        _mp_detector = _mp_face_detection.FaceDetection(
            model_selection=1,        # 0 = short-range, 1 = full-range
            min_detection_confidence=0.5,
        )
    return _mp_detector


# ── Haar Cascade ──────────────────────────────────────────────────────────────

_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
_haar_cascade = cv2.CascadeClassifier(_CASCADE_PATH)


# ── Public API ────────────────────────────────────────────────────────────────

def detect_face(
    frame_bgr: np.ndarray,
    padding: float = 0.3,
) -> tuple[Optional[np.ndarray], Optional[tuple], float, str]:
    """
    Detect the largest face and return a padded crop.

    Parameters
    ----------
    frame_bgr : BGR uint8 image from OpenCV.
    padding   : Fractional expansion around the detected bounding box.

    Returns
    -------
    crop       : RGB uint8 face crop, or None if nothing found.
    bbox       : (x1, y1, x2, y2) in original image coords, or None.
    confidence : Detection confidence [0, 1].  0 for centre-crop fallback.
    method     : "mediapipe" | "haar" | "center_crop" | "none"
    """
    H, W = frame_bgr.shape[:2]

    # === Tier 1: MediaPipe ===
    crop, bbox, conf = _try_mediapipe(frame_bgr, padding)
    if crop is not None:
        return crop, bbox, conf, "mediapipe"

    # === Tier 2: Haar Cascade ===
    crop, bbox, conf = _try_haar(frame_bgr, padding)
    if crop is not None:
        return crop, bbox, conf, "haar"

    # === Tier 3: Centre crop (last resort) ===
    crop, bbox = _centre_crop(frame_bgr, padding=0.0)
    logger.warning("Face detection failed — using centre crop (low confidence)")
    return crop, bbox, 0.0, "center_crop"


def _pad_bbox(x, y, w, h, padding, H, W):
    """Apply padding to a bounding box and clip to image bounds."""
    pad_w, pad_h = int(w * padding), int(h * padding)
    x1 = max(0, x - pad_w)
    y1 = max(0, y - pad_h)
    x2 = min(W, x + w + pad_w)
    y2 = min(H, y + h + pad_h)
    return x1, y1, x2, y2


def _try_mediapipe(
    frame_bgr: np.ndarray, padding: float,
) -> tuple[Optional[np.ndarray], Optional[tuple], float]:
    """Attempt MediaPipe face detection."""
    detector = _get_mp_detector()
    if detector is None:
        return None, None, 0.0

    H, W = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    results = detector.process(rgb)

    if not results.detections:
        return None, None, 0.0

    # Pick the highest-confidence detection
    best = max(results.detections, key=lambda d: d.score[0])
    bbox_rel = best.location_data.relative_bounding_box
    x = int(bbox_rel.xmin * W)
    y = int(bbox_rel.ymin * H)
    w = int(bbox_rel.width * W)
    h = int(bbox_rel.height * H)
    conf = float(best.score[0])

    x1, y1, x2, y2 = _pad_bbox(x, y, w, h, padding, H, W)
    crop = cv2.cvtColor(frame_bgr[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
    return crop, (x1, y1, x2, y2), conf


def _try_haar(
    frame_bgr: np.ndarray, padding: float,
) -> tuple[Optional[np.ndarray], Optional[tuple], float]:
    """Attempt Haar cascade face detection."""
    H, W = frame_bgr.shape[:2]
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = _haar_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        return None, None, 0.0

    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    x1, y1, x2, y2 = _pad_bbox(x, y, w, h, padding, H, W)
    crop = cv2.cvtColor(frame_bgr[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
    # Haar doesn't provide confidence — use a fixed moderate value
    return crop, (x1, y1, x2, y2), 0.7


def _centre_crop(
    frame_bgr: np.ndarray, padding: float = 0.0,
) -> tuple[np.ndarray, tuple]:
    """Centre crop the image (fallback when no face is found)."""
    H, W = frame_bgr.shape[:2]
    size = min(H, W)
    x1 = (W - size) // 2
    y1 = (H - size) // 2
    x2 = x1 + size
    y2 = y1 + size
    crop = cv2.cvtColor(frame_bgr[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
    return crop, (x1, y1, x2, y2)


# ── Statistics tracker ────────────────────────────────────────────────────────

class FaceDetectionStats:
    """Track face detection success/failure rates across frames."""

    def __init__(self):
        self.total = 0
        self.by_method = {"mediapipe": 0, "haar": 0, "center_crop": 0, "none": 0}

    def record(self, method: str):
        self.total += 1
        self.by_method[method] = self.by_method.get(method, 0) + 1

    @property
    def failure_rate(self) -> float:
        if self.total == 0:
            return 0.0
        failures = self.by_method.get("center_crop", 0) + self.by_method.get("none", 0)
        return failures / self.total

    def summary(self) -> dict:
        return {
            "total_frames": self.total,
            "by_method": dict(self.by_method),
            "failure_rate": round(self.failure_rate, 4),
        }
