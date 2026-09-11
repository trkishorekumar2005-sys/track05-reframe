"""Face tracking per CONTRACT §6 (Hungarian IoU matching)."""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.optimize import linear_sum_assignment

from reframe.config import Config
from reframe.detect import Detection


def _iou(box_a: list[float], box_b: list[float]) -> float:
    """Intersection-over-union of two [x, y, w, h] boxes."""
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


def _make_sample(t: float, det: Detection) -> dict:
    return {
        "t": round(float(t), 3),
        "bbox": list(det.bbox),
        "det_conf": float(det.conf),
        "mouth_open": None,
        "score": None,
    }


class _Track:
    """Internal mutable track state (active or finished)."""

    __slots__ = ("track_id", "samples", "hits", "last_matched_t")

    def __init__(self, track_id: int, t: float, det: Detection):
        self.track_id = track_id
        self.samples: list[dict] = [_make_sample(t, det)]
        self.hits = 1
        self.last_matched_t = float(t)

    @property
    def last_bbox(self) -> list[float]:
        return self.samples[-1]["bbox"]

    @property
    def first_t(self) -> float:
        return self.samples[0]["t"]

    @property
    def last_t(self) -> float:
        return self.samples[-1]["t"]


class IoUTracker:
    """Hungarian IoU-matching multi-face tracker per CONTRACT §6.

    Tracks are confirmed once matched `tracking.min_hits` times, and end once
    unmatched for longer than `tracking.max_missed_s`; a face reappearing
    after that always starts a brand new track id.
    """

    def __init__(self, cfg: Config):
        self._iou_match = cfg.tracking.iou_match
        self._min_hits = cfg.tracking.min_hits
        self._max_missed_s = cfg.tracking.max_missed_s
        self._active: list[_Track] = []
        self._finished: list[_Track] = []
        self._next_id = 1

    def update(self, t: float, detections: list[Detection]) -> None:
        """Match `detections` (one tick) against active tracks and update state."""
        t = float(t)

        still_active: list[_Track] = []
        for track in self._active:
            if t - track.last_matched_t > self._max_missed_s:
                self._finished.append(track)
            else:
                still_active.append(track)
        self._active = still_active

        n_tracks = len(self._active)
        n_dets = len(detections)
        matched_det_idx: set[int] = set()

        if n_tracks > 0 and n_dets > 0:
            iou_matrix = np.zeros((n_tracks, n_dets))
            for i, track in enumerate(self._active):
                for j, det in enumerate(detections):
                    iou_matrix[i, j] = _iou(track.last_bbox, det.bbox)

            row_idx, col_idx = linear_sum_assignment(1.0 - iou_matrix)
            for i, j in zip(row_idx, col_idx):
                if iou_matrix[i, j] >= self._iou_match:
                    track = self._active[i]
                    track.samples.append(_make_sample(t, detections[j]))
                    track.hits += 1
                    track.last_matched_t = t
                    matched_det_idx.add(int(j))

        for j, det in enumerate(detections):
            if j not in matched_det_idx:
                self._active.append(_Track(self._next_id, t, det))
                self._next_id += 1

    def finalize(self) -> list[dict]:
        """Flush remaining active tracks and return confirmed tracks (hits >= min_hits)."""
        self._finished.extend(self._active)
        self._active = []

        confirmed = [tr for tr in self._finished if tr.hits >= self._min_hits]
        confirmed.sort(key=lambda tr: tr.track_id)

        return [
            {
                "track_id": tr.track_id,
                "first_t": tr.first_t,
                "last_t": tr.last_t,
                "samples": tr.samples,
            }
            for tr in confirmed
        ]


def box_at(track: dict, t: float) -> Optional[list[float]]:
    """Linearly interpolate a track's bbox at time `t`.

    Returns None if `t` is outside [first_t, last_t], or falls inside a
    "gap" — an interval between two consecutive samples that is
    noticeably longer than the track's own nominal (shortest observed)
    sample spacing, i.e. one or more ticks were missed there.
    """
    samples = track["samples"]
    if not samples:
        return None

    t = float(t)
    if t < samples[0]["t"] or t > samples[-1]["t"]:
        return None

    for s in samples:
        if abs(s["t"] - t) < 1e-9:
            return list(s["bbox"])

    deltas = [samples[i + 1]["t"] - samples[i]["t"] for i in range(len(samples) - 1)]
    nominal = min(deltas) if deltas else 0.0

    for i in range(len(samples) - 1):
        t0, t1 = samples[i]["t"], samples[i + 1]["t"]
        if t0 <= t <= t1:
            dt = t1 - t0
            if nominal > 0.0 and dt > nominal * 1.5:
                return None
            frac = (t - t0) / dt if dt > 0.0 else 0.0
            b0, b1 = samples[i]["bbox"], samples[i + 1]["bbox"]
            return [b0[k] + (b1[k] - b0[k]) * frac for k in range(4)]

    return None


__all__ = ["IoUTracker", "box_at"]
