"""Evaluation metrics per CONTRACT §14.

Every function here is a pure function of plain dicts/lists (ticks, tracks,
crops, intervals) so it can be unit-tested on tiny hand-made examples without
building a full `Timeline`. `report.py` adapts a loaded `Timeline` + `Label`
into these plain shapes and calls these functions.

Shapes used throughout:
    decision: {"t": float, "mode": "speaker"|"fallback", "track_id": int|None, ...}
    track:    {"samples": [{"t": float, "bbox": [x, y, w, h]}, ...]}
    tracks:   dict[track_id, track]
    interval: {"start": float, "end": float, "expect": str}  # str is "FALLBACK" or a person key
    people:   dict[person_key, (lo, hi)]                     # x_range fractions of width
    crop:     {"f": int, "t": float, "x": int, "y": int, "w": int, "h": int, "mode": str}
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from reframe.crop import check_crop_invariants
from reframe.track import box_at

# CONTRACT §14: a switch is confirmed once the correct target holds
# continuously for this long. This is a fixed part of the metric's own
# definition (like the interval `tolerance_s` is), not a tunable runtime
# threshold, so it is not read from configs/default.yaml.
SWITCH_HOLD_S = 0.5


def _mean(values: list[Optional[float]]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def selected_centre_frac(decision: dict, tracks: dict[int, dict], W: float) -> Optional[float]:
    """centre_x / W of the track framed by `decision`, or None if not a speaker
    decision, the track is unknown, or its box is unavailable at this instant
    (ended track / tracking gap -- see `track.box_at`)."""
    if decision.get("mode") != "speaker" or W <= 0:
        return None
    track_id = decision.get("track_id")
    if track_id is None:
        return None
    track = tracks.get(track_id)
    if track is None:
        return None
    box = box_at(track, decision["t"])
    if box is None:
        return None
    cx = box[0] + box[2] / 2.0
    return cx / W


def _tick_matches_expect(
    decision: dict,
    expect: str,
    people: dict[str, tuple[float, float]],
    tracks: dict[int, dict],
    W: float,
) -> bool:
    """Whether `decision` frames the labelled target `expect` at its own tick."""
    if expect == "FALLBACK":
        return decision.get("mode") == "fallback"
    if decision.get("mode") != "speaker":
        return False
    frac = selected_centre_frac(decision, tracks, W)
    if frac is None:
        return False
    person = people.get(expect)
    if person is None:
        return False
    lo, hi = person
    return lo <= frac <= hi


def visible_speaker_accuracy(
    decisions: list[dict],
    tracks: dict[int, dict],
    intervals: list[dict],
    people: dict[str, tuple[float, float]],
    tolerance_s: float,
    W: float,
) -> Optional[dict]:
    """CONTRACT §14 visible_speaker_accuracy.

    Ticks in each non-FALLBACK interval's [start + tolerance_s, end) window are
    "eligible"; correct if mode == "speaker" and the selected track's box
    centre_x/W falls inside that person's x_range. Returns None when there are
    no eligible ticks at all (e.g. a label with only FALLBACK intervals).
    """
    correct = 0
    total = 0
    for iv in intervals:
        expect = iv["expect"]
        if expect == "FALLBACK":
            continue
        window_start = iv["start"] + tolerance_s
        window_end = iv["end"]
        if window_start >= window_end:
            continue
        for d in decisions:
            t = d["t"]
            if window_start <= t < window_end:
                total += 1
                if _tick_matches_expect(d, expect, people, tracks, W):
                    correct += 1

    if total == 0:
        return None
    return {"correct": correct, "total": total, "value": correct / total}


def fallback_prf(
    decisions: list[dict],
    intervals: list[dict],
    tolerance_s: float,
) -> Optional[dict]:
    """CONTRACT §14 fallback precision/recall/f1 (positive class = FALLBACK) and
    confident_wrong_crop_rate (labelled-FALLBACK ticks framed as speaker).
    Tolerance windows are excluded from every interval. Returns None if no
    ticks fall in any interval window."""
    tp = fp = fn = tn = 0
    for iv in intervals:
        window_start = iv["start"] + tolerance_s
        window_end = iv["end"]
        if window_start >= window_end:
            continue
        positive = iv["expect"] == "FALLBACK"
        for d in decisions:
            t = d["t"]
            if not (window_start <= t < window_end):
                continue
            predicted_positive = d.get("mode") == "fallback"
            if positive and predicted_positive:
                tp += 1
            elif positive and not predicted_positive:
                fn += 1
            elif not positive and predicted_positive:
                fp += 1
            else:
                tn += 1

    n = tp + fp + fn + tn
    if n == 0:
        return None

    precision = tp / (tp + fp) if (tp + fp) > 0 else None
    recall = tp / (tp + fn) if (tp + fn) > 0 else None
    if precision is None or recall is None or (precision + recall) == 0:
        f1 = None
    else:
        f1 = 2 * precision * recall / (precision + recall)
    n_labelled_fallback = tp + fn
    confident_wrong_crop_rate = fn / n_labelled_fallback if n_labelled_fallback > 0 else None

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confident_wrong_crop_rate": confident_wrong_crop_rate,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def switch_latency(
    decisions: list[dict],
    intervals: list[dict],
    people: dict[str, tuple[float, float]],
    tracks: dict[int, dict],
    W: float,
    dt: float,
    hold_s: float = SWITCH_HOLD_S,
) -> dict:
    """CONTRACT §14 switch_latency_median_s / p90_s / switches_missed.

    For every interval after the first, latency is measured from its start T
    to the first tick from which the correct target holds continuously
    (consecutive accumulated tick time, cf. CONTRACT §8) for >= hold_s; an
    interval where no such run starts before its own end counts as missed.
    """
    decisions_sorted = sorted(decisions, key=lambda d: d["t"])
    latencies: list[float] = []
    missed = 0

    for iv in intervals[1:]:
        expect = iv["expect"]
        T = iv["start"]
        iv_end = iv["end"]

        run_start: Optional[float] = None
        accumulated = 0.0
        found: Optional[float] = None

        for d in decisions_sorted:
            t = d["t"]
            if t < T:
                continue
            if _tick_matches_expect(d, expect, people, tracks, W):
                if run_start is None:
                    run_start = t
                    accumulated = dt
                else:
                    accumulated += dt
                if accumulated >= hold_s:
                    found = run_start
                    break
            else:
                run_start = None
                accumulated = 0.0

        if found is None or found >= iv_end:
            missed += 1
        else:
            latencies.append(found - T)

    result: dict = {"switches_missed": missed, "n_intervals_checked": len(intervals) - 1}
    if latencies:
        arr = np.array(latencies, dtype=float)
        result["median_s"] = float(np.median(arr))
        result["p90_s"] = float(np.percentile(arr, 90))
    else:
        result["median_s"] = None
        result["p90_s"] = None
    return result


def crop_bound_violations(crops: list[dict], W: int, H: int, aspect: str) -> dict:
    """CONTRACT §14 crop_bound_violations (count, % frames), recomputed purely
    from the recorded crops via `crop.check_crop_invariants` (no face_box, so
    only the geometric checks -- out_of_frame/odd_dims/wrong_aspect/
    bad_fallback_rect -- apply; face_centre_outside needs the original face
    box, not just the crop)."""
    count = 0
    for c in crops:
        rect = (c["x"], c["y"], c["w"], c["h"])
        codes = check_crop_invariants(rect, W, H, aspect, c["mode"])
        if codes:
            count += 1
    total = len(crops)
    pct = (count / total * 100.0) if total else 0.0
    return {"count": count, "pct": pct}


def _frame_tick_key(
    t_f: float, start_s: float, sample_fps: float, decisions: list[dict]
) -> tuple[str, Optional[int]]:
    """Which decision tick an output frame time falls under, per the same
    formula as `crop.plan_crops` (CONTRACT §9): floor((t_f - start_s) * sample_fps),
    clamped to the decisions range."""
    idx = math.floor((t_f - start_s) * sample_fps)
    idx = max(0, min(idx, len(decisions) - 1))
    d = decisions[idx]
    return (d["mode"], d.get("track_id"))


def eligible_speaker_runs(
    crops: list[dict],
    decisions: list[dict],
    start_s: float,
    sample_fps: float,
) -> list[list[dict]]:
    """Group crop frames into maximal runs sharing one (mode, track_id) speaker
    target -- an uninterrupted speaker segment with no snap (mode change,
    track change, or scene cut, which `decide.py` always forces to a fallback
    tick, per CONTRACT §8 rule 1) -- and drop each run's leading frame (the
    snap position itself), since CONTRACT §14 jitter/speed are defined "without
    snap". Only mode == "speaker" runs are returned."""
    if not crops or not decisions:
        return []

    crops_sorted = sorted(crops, key=lambda c: c["f"])
    runs: list[list[dict]] = []
    current: list[dict] = []
    prev_key: Optional[tuple] = None

    def _flush() -> None:
        if len(current) > 1 and prev_key is not None and prev_key[0] == "speaker":
            runs.append(current[1:])

    for c in crops_sorted:
        key = _frame_tick_key(c["t"], start_s, sample_fps, decisions)
        if key != prev_key:
            _flush()
            current = []
        current.append(c)
        prev_key = key
    _flush()

    return runs


def jitter_and_speed(runs: list[list[dict]], W: float) -> dict:
    """CONTRACT §14 jitter_px_per_frame2 (mean |x[f+1]-2x[f]+x[f-1]|, and the
    same for y) / jitter_norm (= value/W * 1000) / p95_speed_px_per_frame, all
    computed only over the given (already snap-free, same-segment) runs.

    `jitter_px_per_frame2` is reported as the mean of the x- and y-axis second
    differences (also broken out individually as jitter_x/jitter_y); speed per
    frame step is the Euclidean displacement.
    """
    jerk_x: list[float] = []
    jerk_y: list[float] = []
    speeds: list[float] = []

    for run in runs:
        xs = [c["x"] for c in run]
        ys = [c["y"] for c in run]
        for i in range(1, len(xs) - 1):
            jerk_x.append(abs(xs[i + 1] - 2 * xs[i] + xs[i - 1]))
            jerk_y.append(abs(ys[i + 1] - 2 * ys[i] + ys[i - 1]))
        for i in range(len(xs) - 1):
            speeds.append(math.hypot(xs[i + 1] - xs[i], ys[i + 1] - ys[i]))

    jitter_x = _mean(jerk_x)
    jitter_y = _mean(jerk_y)
    jitter_combined = _mean([jitter_x, jitter_y]) if jitter_x is not None and jitter_y is not None else None
    jitter_norm = (jitter_combined / W * 1000.0) if (jitter_combined is not None and W) else None
    p95_speed = float(np.percentile(speeds, 95)) if speeds else None

    return {
        "jitter_x": jitter_x,
        "jitter_y": jitter_y,
        "jitter_px_per_frame2": jitter_combined,
        "jitter_norm": jitter_norm,
        "p95_speed_px_per_frame": p95_speed,
    }


def switches_summary(n_switches: int, duration_s: float) -> dict:
    """CONTRACT §14 n_switches, switches_per_min."""
    minutes = duration_s / 60.0 if duration_s > 0 else None
    per_min = (n_switches / minutes) if minutes else None
    return {"n_switches": n_switches, "switches_per_min": per_min}


__all__ = [
    "SWITCH_HOLD_S",
    "selected_centre_frac",
    "visible_speaker_accuracy",
    "fallback_prf",
    "switch_latency",
    "crop_bound_violations",
    "eligible_speaker_runs",
    "jitter_and_speed",
    "switches_summary",
]
