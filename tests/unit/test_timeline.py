"""Unit tests for the decision_timeline.json schema and I/O per CONTRACT §10."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from reframe.timeline import Timeline, load_timeline, save_timeline


def _sample_dict() -> dict:
    return {
        "schema_version": "1.0",
        "tool": {"name": "reframe", "version": "0.1.0", "git_commit": None},
        "created_utc": "2026-01-01T00:00:00+00:00",
        "input": {
            "path": "tests/fixtures/single_small.mp4",
            "sha256": "a" * 64,
            "size_bytes": 179156,
            "width": 480,
            "height": 864,
            "fps": 29.97,
            "is_vfr": False,
            "duration_s": 2.006,
            "rotation": 0,
            "video_codec": "h264",
            "has_audio": True,
            "audio_codec": "aac",
            "n_frames_expected": 60,
            "warnings": [],
        },
        "target": {"aspect": "9:16", "crop_w": 480, "crop_h": 854, "fps": 29.97},
        "segment": {"start_s": 0.0, "end_s": 2.006},
        "config_hash": "0123456789abcdef",
        "config": {},
        "models": [
            {"name": "blaze_face_short_range", "file": "blaze_face_short_range.tflite", "sha256": "b" * 64, "size_mb": 0.219}
        ],
        "analysis": {
            "sample_fps": 10,
            "analysis_width": 480,
            "analysis_height": 864,
            "detector": "mediapipe",
            "strategy": "largest_face",
            "n_ticks": 20,
        },
        "face_tracks": [
            {
                "track_id": 1,
                "first_t": 0.0,
                "last_t": 0.1,
                "samples": [
                    {"t": 0.0, "bbox": [1.0, 2.0, 3.0, 4.0], "det_conf": 0.9, "mouth_open": None, "score": None},
                    {"t": 0.1, "bbox": [1.0, 2.0, 3.0, 4.0], "det_conf": 0.9, "mouth_open": None, "score": None},
                ],
            }
        ],
        "vad_segments": [{"start": 0.0, "end": 0.5}],
        "scene_cuts": [1.0],
        "decisions": [
            {"t": 0.0, "mode": "speaker", "track_id": 1, "confidence": 0.9, "reason": "largest_face", "scores": {1: 0.9}}
        ],
        "speaker_segments": [
            {"start": 0.0, "end": 0.1, "mode": "speaker", "track_id": 1, "mean_confidence": 0.9, "reason": "largest_face"}
        ],
        "switches": [{"t": 0.0, "from": "FALLBACK", "to": 1, "reason": "largest_face", "confidence": 0.9}],
        "fallback_periods": [],
        "crops": [{"f": 0, "t": 0.0, "x": 0, "y": 0, "w": 480, "h": 854, "mode": "speaker"}],
        "invariant_violations": {"count": 0, "by_code": {}, "examples": []},
        "warnings": [],
        "timing": {"analysis_s": 1.234},
    }


def test_timeline_round_trip(tmp_path: Path):
    """save_timeline then load_timeline reproduces the same data."""
    original = Timeline.model_validate(_sample_dict())
    out_path = tmp_path / "decision_timeline.json"

    save_timeline(original, out_path)
    loaded = load_timeline(out_path)

    assert loaded.model_dump() == original.model_dump()


def test_save_timeline_writes_indent_1_and_forward_slash_paths(tmp_path: Path):
    original = Timeline.model_validate(_sample_dict())
    out_path = tmp_path / "decision_timeline.json"
    save_timeline(original, out_path)

    raw = out_path.read_text(encoding="utf-8")
    data = json.loads(raw)
    assert "/" in data["input"]["path"]
    assert "\\" not in data["input"]["path"]
    # indent=1 means nested keys are indented by exactly 1 space per level.
    assert '\n {\n  "name"' in raw or '\n "tool"' in raw


def test_load_timeline_rejects_missing_required_key(tmp_path: Path):
    """A decision_timeline.json missing a required top-level key fails validation."""
    data = _sample_dict()
    del data["config_hash"]
    p = tmp_path / "broken.json"
    p.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValidationError):
        load_timeline(p)


def test_load_timeline_rejects_missing_nested_key(tmp_path: Path):
    """A missing key nested inside a sub-model is also rejected."""
    data = _sample_dict()
    del data["input"]["sha256"]
    p = tmp_path / "broken_nested.json"
    p.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValidationError):
        load_timeline(p)


def test_timeline_rejects_unknown_top_level_key():
    """extra='forbid' rejects unexpected top-level keys."""
    data = _sample_dict()
    data["unexpected_key"] = 123
    with pytest.raises(ValidationError):
        Timeline.model_validate(data)
