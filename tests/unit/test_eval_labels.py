"""Unit tests for label loading/validation per CONTRACT §14."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from reframe.eval.labels import Label, load_labels


def _valid_dict() -> dict:
    return {
        "clip": "S1_turns.mp4",
        "split": "test",
        "people": {"A": {"x_range": [0.0, 0.5]}, "B": {"x_range": [0.5, 1.0]}},
        "intervals": [
            {"start": 0.0, "end": 5.2, "expect": "A"},
            {"start": 5.2, "end": 11.0, "expect": "B"},
            {"start": 11.0, "end": 15.0, "expect": "FALLBACK"},
        ],
        "tolerance_s": 0.5,
    }


def test_valid_label_parses():
    label = Label.model_validate(_valid_dict())
    assert label.clip == "S1_turns.mp4"
    assert label.people["A"].x_range == (0.0, 0.5)
    assert len(label.intervals) == 3


def test_x_range_must_be_within_0_1():
    data = _valid_dict()
    data["people"]["A"]["x_range"] = [5.0, 23.0]
    with pytest.raises(ValidationError):
        Label.model_validate(data)


def test_x_range_lo_must_be_less_than_hi():
    data = _valid_dict()
    data["people"]["A"]["x_range"] = [0.5, 0.5]
    with pytest.raises(ValidationError):
        Label.model_validate(data)


def test_interval_start_must_be_less_than_end():
    data = _valid_dict()
    data["intervals"][0]["end"] = 0.0
    with pytest.raises(ValidationError):
        Label.model_validate(data)


def test_interval_expect_must_reference_known_person_or_fallback():
    data = _valid_dict()
    data["intervals"][0]["expect"] = "C"
    with pytest.raises(ValidationError):
        Label.model_validate(data)


def test_tolerance_s_must_be_positive():
    data = _valid_dict()
    data["tolerance_s"] = 0.0
    with pytest.raises(ValidationError):
        Label.model_validate(data)


def test_intervals_must_be_nonempty():
    data = _valid_dict()
    data["intervals"] = []
    with pytest.raises(ValidationError):
        Label.model_validate(data)


def test_extra_key_is_rejected():
    data = _valid_dict()
    data["unexpected"] = 1
    with pytest.raises(ValidationError):
        Label.model_validate(data)


def test_fallback_only_label_needs_no_people():
    data = {
        "clip": "F1_offscreen.mp4",
        "split": "test",
        "people": {},
        "intervals": [{"start": 6.0, "end": 32.0, "expect": "FALLBACK"}],
        "tolerance_s": 0.5,
    }
    label = Label.model_validate(data)
    assert label.people == {}


def test_load_labels_valid_directory(tmp_path: Path):
    (tmp_path / "s1.json").write_text(json.dumps(_valid_dict()), encoding="utf-8")
    other = _valid_dict()
    other["clip"] = "F1_offscreen.mp4"
    other["people"] = {}
    other["intervals"] = [{"start": 6.0, "end": 32.0, "expect": "FALLBACK"}]
    (tmp_path / "f1.json").write_text(json.dumps(other), encoding="utf-8")

    result = load_labels(tmp_path)

    assert set(result.by_clip) == {"S1_turns.mp4", "F1_offscreen.mp4"}
    assert result.errors == {}


def test_load_labels_skips_invalid_json_but_keeps_going(tmp_path: Path):
    (tmp_path / "broken.json").write_text("", encoding="utf-8")
    (tmp_path / "ok.json").write_text(json.dumps(_valid_dict()), encoding="utf-8")

    result = load_labels(tmp_path)

    assert "S1_turns.mp4" in result.by_clip
    assert "broken.json" in result.errors


def test_load_labels_skips_schema_invalid_file(tmp_path: Path):
    bad = _valid_dict()
    bad["people"]["A"]["x_range"] = [5.0, 23.0]
    (tmp_path / "bad.json").write_text(json.dumps(bad), encoding="utf-8")

    result = load_labels(tmp_path)

    assert result.by_clip == {}
    assert "bad.json" in result.errors


def test_load_labels_flags_duplicate_clip(tmp_path: Path):
    (tmp_path / "a.json").write_text(json.dumps(_valid_dict()), encoding="utf-8")
    (tmp_path / "b.json").write_text(json.dumps(_valid_dict()), encoding="utf-8")

    result = load_labels(tmp_path)

    assert "S1_turns.mp4" in result.by_clip
    assert "b.json" in result.errors  # a.json sorts first, wins; b.json flagged


def test_load_labels_missing_directory_returns_empty(tmp_path: Path):
    result = load_labels(tmp_path / "does_not_exist")
    assert result.by_clip == {}
    assert result.errors == {}
