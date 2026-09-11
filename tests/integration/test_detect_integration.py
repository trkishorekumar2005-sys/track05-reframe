"""Integration test: face detection on tests/fixtures/single_small.mp4 (CONTRACT §2)."""

from pathlib import Path

import pytest

from reframe.config import load_config
from reframe.detect import make_detector
from reframe.errors import MissingDependencyError
from reframe.ffmpeg_utils import FrameReader
from reframe.probe import probe

FIXTURE = Path(__file__).resolve().parent.parent / "fixtures" / "single_small.mp4"


def test_face_found_in_most_sampled_frames():
    """A face is detected in >= 80% of sampled analysis frames of single_small.mp4."""
    cfg = load_config()
    info = probe(FIXTURE, cfg)

    out_w = min(cfg.analysis.analysis_width, info.width)
    scale = out_w / info.width
    out_h = int(round(info.height * scale))
    if out_h % 2:
        out_h += 1

    try:
        detector = make_detector(cfg, scale)
    except MissingDependencyError:
        pytest.skip(
            "Model files missing/invalid; run `uv run python scripts/download_models.py` first."
        )

    n_ticks = 0
    n_with_face = 0
    try:
        with FrameReader(
            FIXTURE,
            out_w=out_w,
            out_h=out_h,
            fps=cfg.analysis.sample_fps,
            start_s=0.0,
            end_s=info.duration_s,
            mode="analysis",
        ) as reader:
            for _, t, frame in reader:
                t_ms = int(round(t * 1000))
                detections = detector.detect(frame, t_ms)
                n_ticks += 1
                if detections:
                    n_with_face += 1
    finally:
        detector.close()

    assert n_ticks > 0
    assert n_with_face / n_ticks >= 0.8
