"""Scene analysis per CONTRACT §2, §7: cut detection and screen-content detection.

cv2 is used here for its exact, standard implementations of HSV histograms,
Bhattacharyya distance and Canny edge detection (it is an unavoidable locked
transitive dependency of mediapipe, not something added on top of it); it is
never used for decoding (that stays FrameReader-only per CONTRACT §3).
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from reframe.config import Config

_HIST_BINS = (50, 60)  # (hue, saturation)
_CANNY_LOW, _CANNY_HIGH = 100, 200


def _hsv_histogram(frame_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, list(_HIST_BINS), [0, 180, 0, 256])
    cv2.normalize(hist, hist, alpha=1, norm_type=cv2.NORM_L1)
    return hist


def _canny_edge_density(frame_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, _CANNY_LOW, _CANNY_HIGH)
    return float(np.count_nonzero(edges)) / float(edges.size)


class SceneAnalyzer:
    """Stateful per-tick scene-cut and screen-content detector per CONTRACT §7.

    Ticks must be presented in increasing time order (one call per tick);
    `is_cut` compares each frame's HSV histogram to the previous tick's.
    """

    def __init__(self, cfg: Config) -> None:
        self._enabled = cfg.scene.enabled
        self._cut_threshold = cfg.scene.cut_threshold
        self._screen_edge_density = cfg.scene.screen_edge_density
        self._prev_hist: Optional[np.ndarray] = None

    def analyze(self, frame_bgr: np.ndarray, n_faces: int) -> tuple[bool, bool]:
        """Returns (is_cut, is_screen_content) for this tick's frame."""
        if not self._enabled:
            return False, False

        hist = _hsv_histogram(frame_bgr)
        is_cut = False
        if self._prev_hist is not None:
            dist = cv2.compareHist(self._prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA)
            is_cut = dist > self._cut_threshold
        self._prev_hist = hist

        is_screen_content = False
        if n_faces == 0:
            edge_density = _canny_edge_density(frame_bgr)
            is_screen_content = edge_density > self._screen_edge_density

        return is_cut, is_screen_content


__all__ = ["SceneAnalyzer"]
