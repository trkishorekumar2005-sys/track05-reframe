"""Unit tests for score.py per CONTRACT §7: av_heuristic / av_corr vs largest_face.

Scenario: a large, still-mouthed face and a small, moving-mouthed face, both
visible throughout continuous speech. largest_face (area-only) should lock
onto the large face; av_heuristic and av_corr (both audio/lip-driven) should
lock onto the small, actually-talking face instead.
"""

from reframe.audio import AudioFeatures
from reframe.config import Config
from reframe.decide import decide
from reframe.score import score_ticks

_SAMPLE_FPS = 10.0
_N_TICKS = 30

_LARGE_ID = 1
_SMALL_ID = 2


def _tick_times(n: int) -> list[float]:
    return [round(k / _SAMPLE_FPS, 3) for k in range(n)]


def _make_track(track_id: int, bbox: list[float], mouth_series: list[float], tick_times: list[float]) -> dict:
    return {
        "track_id": track_id,
        "first_t": tick_times[0],
        "last_t": tick_times[-1],
        "samples": [
            {"t": t, "bbox": list(bbox), "det_conf": 0.9, "mouth_open": m, "score": None}
            for t, m in zip(tick_times, mouth_series)
        ],
    }


def _build_scenario():
    tick_times = _tick_times(_N_TICKS)

    large_bbox = [0.0, 0.0, 300.0, 300.0]  # area 90000
    small_bbox = [400.0, 400.0, 50.0, 50.0]  # area 2500

    still_mouth = [0.1] * _N_TICKS
    moving_mouth = [0.9 if k % 2 == 0 else 0.1 for k in range(_N_TICKS)]

    tracks = [
        _make_track(_LARGE_ID, large_bbox, still_mouth, tick_times),
        _make_track(_SMALL_ID, small_bbox, moving_mouth, tick_times),
    ]

    # Energy in sync with the small track's mouth: strong positive av_corr for
    # it, while the large track's constant mouth forces its av_corr to 0
    # regardless of energy (std < 1e-6 per CONTRACT §7).
    energy = [0.9 if k % 2 == 0 else 0.1 for k in range(_N_TICKS)]
    vad = [1.0] * _N_TICKS  # continuous speech throughout

    audio = AudioFeatures(
        vad=vad,
        energy=energy,
        vad_segments=[{"start": tick_times[0], "end": tick_times[-1]}],
    )
    return tracks, tick_times, audio


def _run_pipeline(strategy: str):
    tracks, tick_times, audio = _build_scenario()
    cfg = Config()

    scores_per_tick = score_ticks(tracks, tick_times, strategy, cfg, audio=audio)
    ticks = [
        {
            "t": tick_times[i],
            "visible_ids": list(scores_per_tick[i].keys()),
            "scores": scores_per_tick[i],
            "vad": audio.vad[i],
            "is_cut": False,
            "is_screen_content": False,
        }
        for i in range(len(tick_times))
    ]
    decisions, _ = decide(ticks, cfg, has_audio=True)
    return decisions, scores_per_tick


def test_score_values_favor_the_moving_small_face_under_av_heuristic():
    _, scores_per_tick = _run_pipeline("av_heuristic")
    last = scores_per_tick[-1]
    assert last[_SMALL_ID] > last[_LARGE_ID]
    assert last[_LARGE_ID] == 0.0  # constant mouth -> lip_activity 0 -> score 0


def test_score_values_favor_the_moving_small_face_under_av_corr():
    _, scores_per_tick = _run_pipeline("av_corr")
    last = scores_per_tick[-1]
    assert last[_SMALL_ID] > last[_LARGE_ID]
    assert last[_LARGE_ID] == 0.0  # constant mouth -> av_corr forced to 0


def test_largest_face_locks_onto_the_large_still_face():
    decisions, _ = _run_pipeline("largest_face")
    final = decisions[-1]
    assert final["mode"] == "speaker"
    assert final["track_id"] == _LARGE_ID


def test_av_heuristic_locks_onto_the_small_moving_face():
    decisions, _ = _run_pipeline("av_heuristic")
    final = decisions[-1]
    assert final["mode"] == "speaker"
    assert final["track_id"] == _SMALL_ID


def test_av_corr_locks_onto_the_small_moving_face():
    decisions, _ = _run_pipeline("av_corr")
    final = decisions[-1]
    assert final["mode"] == "speaker"
    assert final["track_id"] == _SMALL_ID
