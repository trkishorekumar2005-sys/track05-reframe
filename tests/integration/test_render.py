"""Integration test: render() per CONTRACT §11 on a hand-written Timeline."""

import subprocess
from pathlib import Path

import pytest

from reframe import __version__
from reframe.config import Config, CropConfig
from reframe.crop import crop_size
from reframe.probe import probe
from reframe.render import render
from reframe.timeline import (
    AnalysisInfo,
    SegmentInfo,
    TargetInfo,
    Timeline,
    TimingInfo,
    ToolInfo,
    save_timeline,
)


@pytest.fixture
def clip(tmp_path: Path) -> Path:
    """3s 1280x720 30fps testsrc + sine clip with audio."""
    out_file = tmp_path / "input.mp4"
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "testsrc=duration=3:size=1280x720:rate=30",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=440:duration=3",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        out_file.as_posix(),
    ]
    subprocess.run(cmd, check=True, capture_output=True, timeout=30)
    return out_file


def _audio_md5(path: Path) -> str:
    cmd = ["ffmpeg", "-v", "error", "-i", path.as_posix(), "-map", "0:a", "-f", "md5", "-"]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=True)
    return res.stdout.strip()


def _build_timeline(info) -> Timeline:
    W, H, fps = info.width, info.height, info.fps
    crop_w, crop_h = crop_size(W, H, "9:16")
    n_frames = round(info.duration_s * fps)
    x = (W - crop_w) // 2

    crops = []
    for f in range(n_frames):
        t = round(f / fps, 3)
        if f < n_frames // 2:
            crops.append(
                {"f": f, "t": t, "x": x, "y": 0, "w": crop_w, "h": crop_h, "mode": "speaker"}
            )
        else:
            crops.append({"f": f, "t": t, "x": 0, "y": 0, "w": W, "h": H, "mode": "fallback"})

    return Timeline(
        tool=ToolInfo(name="reframe", version=__version__, git_commit=None),
        created_utc="2026-01-01T00:00:00+00:00",
        input=info,
        target=TargetInfo(aspect="9:16", crop_w=crop_w, crop_h=crop_h, fps=fps),
        segment=SegmentInfo(start_s=0.0, end_s=info.duration_s),
        config_hash="0" * 16,
        config=Config(crop=CropConfig(aspect="9:16")),
        models=[],
        analysis=AnalysisInfo(
            sample_fps=10,
            analysis_width=640,
            analysis_height=360,
            detector="mediapipe",
            strategy="largest_face",
            n_ticks=0,
        ),
        face_tracks=[],
        vad_segments=[],
        scene_cuts=[],
        decisions=[],
        speaker_segments=[],
        switches=[],
        fallback_periods=[],
        crops=crops,
        invariant_violations={"count": 0, "by_code": {}, "examples": []},
        warnings=[],
        timing=TimingInfo(analysis_s=0.0),
    )


def test_render_half_speaker_half_fallback(tmp_path: Path, clip: Path):
    info = probe(clip)
    timeline = _build_timeline(info)

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    timeline_path = run_dir / "decision_timeline.json"
    save_timeline(timeline, timeline_path)

    out_path = run_dir / "output.mp4"
    result_path = render(timeline_path, out_path, Config())
    assert result_path == out_path
    assert out_path.is_file()

    probed = probe(out_path)
    assert (probed.width, probed.height) == (404, 720)

    expected_duration = timeline.segment.end_s - timeline.segment.start_s
    tol = max(2.0 / timeline.target.fps, 0.1)
    assert abs(probed.duration_s - expected_duration) <= tol

    assert probed.has_audio is True

    md5_in = _audio_md5(clip)
    md5_out = _audio_md5(out_path)
    assert md5_in == md5_out
