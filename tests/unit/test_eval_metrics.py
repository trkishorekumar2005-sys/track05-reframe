"""Unit tests for evaluation metrics per CONTRACT §14, on tiny hand-made
decisions/tracks/crops with known answers."""

import numpy as np

from reframe.eval.metrics import (
    crop_bound_violations,
    eligible_speaker_runs,
    fallback_prf,
    jitter_and_speed,
    selected_centre_frac,
    switch_latency,
    switches_summary,
    visible_speaker_accuracy,
)


def _speaker_tick(t: float, track_id: int) -> dict:
    return {"t": t, "mode": "speaker", "track_id": track_id}


def _fallback_tick(t: float) -> dict:
    return {"t": t, "mode": "fallback", "track_id": None}


# ---------------------------------------------------------------------------
# visible_speaker_accuracy
# ---------------------------------------------------------------------------


def test_visible_speaker_accuracy_10_ticks_7_correct():
    """7 of 10 ticks have the selected track's box centre inside A's x_range."""
    ticks = [round(0.1 * k, 3) for k in range(10)]
    correct_ts = set(ticks[:7])  # first 7 ticks correct, last 3 wrong
    decisions = [_speaker_tick(t, 1) for t in ticks]

    samples = []
    for t in ticks:
        x = 0.0 if t in correct_ts else 90.0  # centre 5/100=0.05 (in A) vs 95/100=0.95 (not in A)
        samples.append({"t": t, "bbox": [x, 0.0, 10.0, 10.0]})
    tracks = {1: {"samples": samples}}

    intervals = [{"start": 0.0, "end": 1.0, "expect": "A"}]
    people = {"A": (0.0, 0.5)}

    result = visible_speaker_accuracy(decisions, tracks, intervals, people, tolerance_s=0.0, W=100.0)

    assert result == {"correct": 7, "total": 10, "value": 0.7}


def test_visible_speaker_accuracy_excludes_tolerance_window():
    ticks = [round(0.1 * k, 3) for k in range(10)]
    decisions = [_speaker_tick(t, 1) for t in ticks]
    samples = [{"t": t, "bbox": [0.0, 0.0, 10.0, 10.0]} for t in ticks]  # always correct
    tracks = {1: {"samples": samples}}

    intervals = [{"start": 0.0, "end": 1.0, "expect": "A"}]
    people = {"A": (0.0, 0.5)}

    result = visible_speaker_accuracy(decisions, tracks, intervals, people, tolerance_s=0.31, W=100.0)

    # ticks < 0.31 excluded: 0.0..0.3 (4 ticks) excluded, 0.4..0.9 remain (6 ticks)
    assert result == {"correct": 6, "total": 6, "value": 1.0}


def test_visible_speaker_accuracy_none_when_only_fallback_intervals():
    intervals = [{"start": 0.0, "end": 1.0, "expect": "FALLBACK"}]
    result = visible_speaker_accuracy([], {}, intervals, {}, tolerance_s=0.0, W=100.0)
    assert result is None


def test_selected_centre_frac_none_for_fallback_decision():
    assert selected_centre_frac(_fallback_tick(0.0), {}, W=100.0) is None


# ---------------------------------------------------------------------------
# fallback_prf
# ---------------------------------------------------------------------------


def test_fallback_prf_known_confusion_matrix():
    # interval 1: labelled FALLBACK, 5 ticks -> 4 predicted fallback (TP), 1 predicted speaker (FN)
    fb_ticks = [_fallback_tick(t) for t in (0.0, 0.1, 0.2, 0.3)] + [_speaker_tick(0.4, 1)]
    # interval 2: labelled speaker "A", 5 ticks -> 3 predicted speaker (TN), 2 predicted fallback (FP)
    sp_ticks = [_speaker_tick(t, 1) for t in (0.5, 0.6, 0.7)] + [_fallback_tick(0.8), _fallback_tick(0.9)]
    decisions = fb_ticks + sp_ticks

    intervals = [
        {"start": 0.0, "end": 0.5, "expect": "FALLBACK"},
        {"start": 0.5, "end": 1.0, "expect": "A"},
    ]

    result = fallback_prf(decisions, intervals, tolerance_s=0.0)

    assert result["tp"] == 4
    assert result["fn"] == 1
    assert result["fp"] == 2
    assert result["tn"] == 3
    assert result["precision"] == 4 / 6
    assert result["recall"] == 4 / 5
    assert abs(result["f1"] - (2 * (4 / 6) * (4 / 5) / ((4 / 6) + (4 / 5)))) < 1e-9
    assert result["confident_wrong_crop_rate"] == 1 / 5


def test_fallback_prf_none_when_no_ticks_in_window():
    intervals = [{"start": 0.0, "end": 0.1, "expect": "FALLBACK"}]
    result = fallback_prf([], intervals, tolerance_s=0.0)
    assert result is None


# ---------------------------------------------------------------------------
# switch_latency
# ---------------------------------------------------------------------------


def test_switch_latency_known_value():
    # Interval 0 is skipped (it's "the first"); interval 1 starts at T=1.0,
    # expects FALLBACK. Ticks 1.0-1.2 are wrong (speaker), 1.3-1.9 are fallback
    # (correct); at dt=0.1 the hold reaches 0.5s after 5 correct ticks, i.e. at
    # t=1.7 -- so the run is confirmed starting from t=1.3 -> latency 0.3s.
    intervals = [
        {"start": 0.0, "end": 1.0, "expect": "A"},
        {"start": 1.0, "end": 5.0, "expect": "FALLBACK"},
    ]
    decisions = (
        [_speaker_tick(round(1.0 + 0.1 * k, 3), 1) for k in range(3)]
        + [_fallback_tick(round(1.3 + 0.1 * k, 3)) for k in range(7)]
    )

    result = switch_latency(decisions, intervals, people={}, tracks={}, W=100.0, dt=0.1)

    assert result["switches_missed"] == 0
    assert abs(result["median_s"] - 0.3) < 1e-9
    assert abs(result["p90_s"] - 0.3) < 1e-9


def test_switch_latency_missed_when_hold_never_reached():
    intervals = [
        {"start": 0.0, "end": 1.0, "expect": "A"},
        {"start": 1.0, "end": 1.5, "expect": "FALLBACK"},
    ]
    # Only 2 correct ticks in a row before the interval ends -> never holds 0.5s.
    decisions = [_fallback_tick(1.0), _fallback_tick(1.1), _speaker_tick(1.2, 1)]

    result = switch_latency(decisions, intervals, people={}, tracks={}, W=100.0, dt=0.1)

    assert result["switches_missed"] == 1
    assert result["median_s"] is None
    assert result["p90_s"] is None


# ---------------------------------------------------------------------------
# crop_bound_violations
# ---------------------------------------------------------------------------


def test_crop_bound_violations_known_count():
    crops = [
        {"f": 0, "t": 0.0, "x": 0, "y": 0, "w": 100, "h": 100, "mode": "fallback"},  # ok
        {"f": 1, "t": 0.1, "x": 60, "y": 0, "w": 50, "h": 50, "mode": "speaker"},  # out_of_frame
        {"f": 2, "t": 0.2, "x": 0, "y": 0, "w": 50, "h": 51, "mode": "speaker"},  # odd_dims + wrong_aspect
        {"f": 3, "t": 0.3, "x": 0, "y": 0, "w": 90, "h": 100, "mode": "fallback"},  # bad_fallback_rect
        {"f": 4, "t": 0.4, "x": 25, "y": 25, "w": 50, "h": 50, "mode": "speaker"},  # ok
    ]

    result = crop_bound_violations(crops, W=100, H=100, aspect="1:1")

    assert result == {"count": 3, "pct": 60.0}


def test_crop_bound_violations_empty_crops():
    result = crop_bound_violations([], W=100, H=100, aspect="1:1")
    assert result == {"count": 0, "pct": 0.0}


# ---------------------------------------------------------------------------
# eligible_speaker_runs
# ---------------------------------------------------------------------------


def test_eligible_speaker_runs_splits_on_track_change_and_drops_snap_frame():
    decisions = (
        [_fallback_tick(0.0)]
        + [_speaker_tick(0.1, 1), _speaker_tick(0.2, 1), _speaker_tick(0.3, 1)]
        + [_speaker_tick(0.4, 2), _speaker_tick(0.5, 2)]
        + [_fallback_tick(0.6)]
        + [_speaker_tick(0.7, 1), _speaker_tick(0.8, 1), _speaker_tick(0.9, 1)]
    )
    crops = [{"f": f, "t": round(0.1 * f, 3), "x": f * 10, "y": 0, "w": 10, "h": 10, "mode": "x"} for f in range(10)]

    runs = eligible_speaker_runs(crops, decisions, start_s=0.0, sample_fps=10.0)

    frame_ids = [[c["f"] for c in run] for run in runs]
    assert frame_ids == [[2, 3], [5], [8, 9]]


def test_eligible_speaker_runs_empty_when_no_crops_or_decisions():
    assert eligible_speaker_runs([], [], 0.0, 10.0) == []
    assert eligible_speaker_runs([{"f": 0, "t": 0.0, "x": 0, "y": 0}], [], 0.0, 10.0) == []


# ---------------------------------------------------------------------------
# jitter_and_speed
# ---------------------------------------------------------------------------


def test_jitter_and_speed_known_values():
    run = [
        {"x": 0, "y": 0},
        {"x": 10, "y": 0},
        {"x": 20, "y": 0},
        {"x": 20, "y": 0},
        {"x": 10, "y": 0},
        {"x": 0, "y": 0},
    ]

    result = jitter_and_speed([run], W=1000.0)

    assert result["jitter_x"] == 5.0
    assert result["jitter_y"] == 0.0
    assert result["jitter_px_per_frame2"] == 2.5
    assert result["jitter_norm"] == 2.5
    assert result["p95_speed_px_per_frame"] == 10.0


def test_jitter_and_speed_empty_runs_returns_none():
    result = jitter_and_speed([], W=1000.0)
    assert result["jitter_px_per_frame2"] is None
    assert result["jitter_norm"] is None
    assert result["p95_speed_px_per_frame"] is None


# ---------------------------------------------------------------------------
# switches_summary
# ---------------------------------------------------------------------------


def test_switches_summary_known_rate():
    result = switches_summary(n_switches=6, duration_s=120.0)
    assert result == {"n_switches": 6, "switches_per_min": 3.0}


def test_switches_summary_zero_duration_is_none():
    result = switches_summary(n_switches=0, duration_s=0.0)
    assert result["switches_per_min"] is None
