"""Mouth-openness extraction per CONTRACT §2, §7.

Uses mediapipe.tasks.python.vision.FaceLandmarker (Tasks API only). Landmarker
faces are matched to tracker boxes by IoU so `mouth_open` can be attached to
the right track's sample for this tick.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    FaceLandmarker as _MPFaceLandmarker,
    FaceLandmarkerOptions,
    RunningMode,
)

from reframe.config import Config
from reframe.errors import ProcessingError
from reframe.models import ensure_model

_MOUTH_MATCH_IOU = 0.3
_JAW_OPEN_BLENDSHAPE = "jawOpen"


def _iou(box_a: list[float], box_b: list[float]) -> float:
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh

    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih

    union = aw * ah + bw * bh - inter
    if union <= 0.0:
        return 0.0
    return inter / union


class MouthAnalyzer:
    """Per-tick mouth-openness (`jawOpen` blendshape), matched to tracker boxes by IoU."""

    def __init__(self, model_path: Path, cfg: Config, scale: float) -> None:
        self._scale = float(scale)
        self._last_t_ms: int = -1

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=RunningMode.VIDEO,
            num_faces=cfg.analysis.max_faces,
            output_face_blendshapes=True,
        )
        self._landmarker = _MPFaceLandmarker.create_from_options(options)

    def analyze(
        self, frame_bgr: np.ndarray, t_ms: int, track_boxes: dict[int, list[float]]
    ) -> dict[int, float]:
        """Return {track_id: mouth_open} for tracks matched to a landmarker face this tick."""
        t_ms = int(t_ms)
        if t_ms <= self._last_t_ms:
            raise ProcessingError(
                f"MouthAnalyzer requires strictly increasing timestamps; "
                f"got {t_ms} after {self._last_t_ms}",
                code="timestamp_error",
            )
        self._last_t_ms = t_ms

        if not track_boxes:
            # Still need to call detect_for_video to keep VIDEO-mode timestamps monotonic.
            frame_rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            self._landmarker.detect_for_video(mp_image, t_ms)
            return {}

        frame_rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])
        h_a, w_a = frame_bgr.shape[:2]
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = self._landmarker.detect_for_video(mp_image, t_ms)

        mouth_by_track: dict[int, float] = {}
        if not result or not result.face_landmarks:
            return mouth_by_track

        inv = 1.0 / self._scale
        for landmarks, blendshapes in zip(result.face_landmarks, result.face_blendshapes):
            xs = [lm.x * w_a for lm in landmarks]
            ys = [lm.y * h_a for lm in landmarks]
            x0, x1 = min(xs), max(xs)
            y0, y1 = min(ys), max(ys)
            bbox_orig = [x0 * inv, y0 * inv, (x1 - x0) * inv, (y1 - y0) * inv]

            jaw_open = 0.0
            for cat in blendshapes:
                if cat.category_name == _JAW_OPEN_BLENDSHAPE:
                    jaw_open = float(cat.score)
                    break

            best_tid: Optional[int] = None
            best_iou = 0.0
            for tid, tbox in track_boxes.items():
                iou = _iou(bbox_orig, tbox)
                if iou >= _MOUTH_MATCH_IOU and iou > best_iou:
                    best_iou = iou
                    best_tid = tid

            if best_tid is not None:
                mouth_by_track[best_tid] = jaw_open

        return mouth_by_track

    def close(self) -> None:
        try:
            self._landmarker.close()
        except Exception:
            pass


def make_mouth_analyzer(cfg: Config, scale: float) -> MouthAnalyzer:
    model_path = ensure_model("face_landmarker")
    return MouthAnalyzer(model_path=model_path, cfg=cfg, scale=scale)


def window_features(
    mouth_series: list[Optional[float]], energy: list[float], k: int, window_ticks: int
) -> tuple[float, float]:
    """Per CONTRACT §7: lip_activity = std(mouth_open) over the window; av_corr = Pearson(mouth, energy).

    The window covers ticks [k - window_ticks + 1, k], skipping ticks where the
    track had no mouth_open sample. av_corr is 0 if either series' std < 1e-6.
    """
    start = max(0, k - window_ticks + 1)
    idx = [i for i in range(start, k + 1) if i < len(mouth_series) and mouth_series[i] is not None]

    if not idx:
        return 0.0, 0.0

    m_vals = np.array([mouth_series[i] for i in idx], dtype=float)
    e_vals = np.array([energy[i] for i in idx], dtype=float)

    lip_activity = float(np.std(m_vals))

    if len(m_vals) < 2:
        return lip_activity, 0.0

    m_std = float(np.std(m_vals))
    e_std = float(np.std(e_vals))
    if m_std < 1e-6 or e_std < 1e-6:
        return lip_activity, 0.0

    av_corr = float(np.corrcoef(m_vals, e_vals)[0, 1])
    return lip_activity, av_corr


__all__ = ["MouthAnalyzer", "make_mouth_analyzer", "window_features"]
