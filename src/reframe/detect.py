"""Face detection per CONTRACT §2, §3.

Uses mediapipe.tasks.python.vision (Tasks API only; mp.solutions is NOT used).
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Protocol, runtime_checkable

import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    FaceDetector as _MPFaceDetector,
    FaceDetectorOptions,
    RunningMode,
)

from reframe.config import Config
from reframe.errors import ProcessingError
from reframe.models import ensure_model


@dataclasses.dataclass
class Detection:
    """A single face detection result.

    Attributes:
        bbox: [x, y, w, h] in ORIGINAL display-pixel coordinates (1 decimal, clipped).
        conf: Detection confidence score in [0, 1].
    """

    bbox: list[float]
    conf: float


@runtime_checkable
class FaceDetector(Protocol):
    """Protocol for face detectors per CONTRACT §2."""

    def detect(self, frame_bgr: np.ndarray, t_ms: int) -> list[Detection]:
        """Detect faces in frame_bgr at timestamp t_ms (milliseconds).

        Args:
            frame_bgr: Analysis-resolution BGR uint8 frame (H, W, 3).
            t_ms: Strictly increasing video timestamp in milliseconds.

        Returns:
            List of Detection objects in original-resolution coordinates.
        """
        ...

    def close(self) -> None:
        """Release underlying detector resources."""
        ...


class MediaPipeDetector:
    """Face detector backed by mediapipe.tasks.python.vision.FaceDetector.

    Contract requirements:
    - Uses VisionTaskRunningMode.VIDEO with strictly-increasing timestamps.
    - Converts BGR input to contiguous RGB before creating mp.Image.
    - Scales bounding boxes by 1/scale to map from analysis-resolution back to
      original-resolution coordinates, then clips to frame bounds.
    - Boxes are [x, y, w, h] rounded to 1 decimal.
    """

    def __init__(self, model_path: Path, cfg: Config, scale: float) -> None:
        self._scale = float(scale)  # analysis_w / original_w (< 1, never upscale)
        self._last_t_ms: int = -1

        options = FaceDetectorOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=RunningMode.VIDEO,
            min_detection_confidence=cfg.analysis.min_face_conf,
        )
        self._detector = _MPFaceDetector.create_from_options(options)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, frame_bgr: np.ndarray, t_ms: int) -> list[Detection]:
        """Detect faces and return results in original-resolution coordinates."""
        t_ms = int(t_ms)
        if t_ms <= self._last_t_ms:
            raise ProcessingError(
                f"MediaPipeDetector requires strictly increasing timestamps; "
                f"got {t_ms} after {self._last_t_ms}",
                code="timestamp_error",
            )
        self._last_t_ms = t_ms

        # Analysis frame is BGR uint8; MediaPipe SRGB expects contiguous RGB.
        frame_rgb: np.ndarray = np.ascontiguousarray(
            frame_bgr[:, :, ::-1], dtype=np.uint8
        )
        h_a, w_a = frame_bgr.shape[:2]

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = self._detector.detect_for_video(mp_image, t_ms)

        detections: list[Detection] = []
        if result and result.detections:
            for det in result.detections:
                bb = det.bounding_box
                # Scale back to original-resolution coords (analysis was downscaled)
                inv = 1.0 / self._scale
                x = bb.origin_x * inv
                y = bb.origin_y * inv
                w = bb.width * inv
                h = bb.height * inv

                # Derive original frame size from analysis frame + scale
                orig_w = w_a / self._scale
                orig_h = h_a / self._scale

                # Clip to frame boundaries
                x = max(0.0, x)
                y = max(0.0, y)
                w = min(w, orig_w - x)
                h = min(h, orig_h - y)

                if w <= 0 or h <= 0:
                    continue

                conf = det.categories[0].score if det.categories else 0.0
                detections.append(
                    Detection(
                        bbox=[round(x, 1), round(y, 1), round(w, 1), round(h, 1)],
                        conf=float(conf),
                    )
                )

        return detections

    def close(self) -> None:
        """Release underlying MediaPipe detector resources."""
        try:
            self._detector.close()
        except Exception:
            pass


def make_detector(cfg: Config, scale: float) -> MediaPipeDetector:
    """Create a MediaPipeDetector for the given Config and analysis scale.

    Args:
        cfg: Loaded reframe Config.
        scale: Ratio analysis_width / original_width (≤ 1, never upscale).

    Returns:
        A ready-to-use MediaPipeDetector.
    """
    model_path = ensure_model("blaze_face_short_range")
    return MediaPipeDetector(model_path=model_path, cfg=cfg, scale=scale)


__all__ = ["Detection", "FaceDetector", "MediaPipeDetector", "make_detector"]
