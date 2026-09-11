"""Integration test: full analyze() pipeline on tests/fixtures/single_small.mp4."""

from pathlib import Path

import pytest

from reframe.analyze import analyze
from reframe.config import load_config
from reframe.errors import MissingDependencyError
from reframe.timeline import load_timeline

FIXTURE = Path(__file__).resolve().parent.parent / "fixtures" / "single_small.mp4"


def _run(tmp_path: Path, name: str):
    cfg = load_config()
    out_dir = tmp_path / name
    try:
        path = analyze(FIXTURE, "9:16", out_dir, cfg)
    except MissingDependencyError:
        pytest.skip(
            "Model files missing/invalid; run `uv run python scripts/download_models.py` first."
        )
    return load_timeline(path)


def test_analyze_produces_valid_timeline_with_no_violations(tmp_path: Path):
    timeline = _run(tmp_path, "run1")

    assert timeline.schema_version == "1.0"

    expected_n_frames = round(
        (timeline.segment.end_s - timeline.segment.start_s) * timeline.input.fps
    )
    assert len(timeline.crops) == expected_n_frames
    assert timeline.invariant_violations.count == 0


def test_analyze_is_deterministic(tmp_path: Path):
    """Two runs on the same input/config produce identical decisions and crops."""
    t1 = _run(tmp_path, "run1")
    t2 = _run(tmp_path, "run2")

    decisions_1 = [d.model_dump() for d in t1.decisions]
    decisions_2 = [d.model_dump() for d in t2.decisions]
    assert decisions_1 == decisions_2

    crops_1 = [c.model_dump() for c in t1.crops]
    crops_2 = [c.model_dump() for c in t2.crops]
    assert crops_1 == crops_2
