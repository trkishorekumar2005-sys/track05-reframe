"""Unit tests for ffmpeg_utils and FrameReader per CONTRACT §3, §4."""

from pathlib import Path
import subprocess
from unittest.mock import patch
import numpy as np
import pytest

from reframe.errors import MissingDependencyError, ProcessingError
from reframe.ffmpeg_utils import FrameReader, require_tools, run_ffmpeg


@pytest.fixture
def sample_clip(tmp_path: Path) -> Path:
    """Generate a 2.0s 320x240 30fps clip with lavfi."""
    clip = tmp_path / "clip.mp4"
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "testsrc=duration=2:size=320x240:rate=30",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        clip.as_posix(),
    ]
    subprocess.run(cmd, check=True, capture_output=True, timeout=15)
    return clip


def test_require_tools_success():
    """require_tools() passes when ffmpeg/ffprobe exist."""
    require_tools()


def test_require_tools_missing():
    """require_tools() raises MissingDependencyError when tool missing."""
    with patch("shutil.which", return_value=None):
        with pytest.raises(MissingDependencyError) as exc_info:
            require_tools()
        assert exc_info.value.code == "ffmpeg_missing"
        assert exc_info.value.exit_code == 3


def test_run_ffmpeg_logging_and_check(tmp_path: Path):
    """run_ffmpeg writes logs and raises on error when check=True."""
    log_file = tmp_path / "ffmpeg.log"
    # Successful run
    res = run_ffmpeg(["-version"], log_file=log_file)
    assert res.returncode == 0

    # Failed run with check=True raises ProcessingError
    with pytest.raises(ProcessingError) as exc_info:
        run_ffmpeg(["-invalid_option_xyz"], log_file=log_file, check=True)
    assert exc_info.value.code == "ffmpeg_error"
    assert exc_info.value.exit_code == 4


def test_framereader_analysis_mode(sample_clip: Path):
    """Analysis mode yields frames with frame count ≈ duration*fps and correct shape."""
    # 2s clip sampled at 10 fps, scaled to 160x120
    reader = FrameReader(sample_clip, out_w=160, out_h=120, fps=10, mode="analysis")
    frames = list(reader)

    # Frame count ≈ duration * fps = 2.0 * 10 = 20
    assert 19 <= len(frames) <= 21

    for idx, (index, t, frame) in enumerate(frames):
        assert index == idx
        assert isinstance(t, float)
        assert 0.0 <= t <= 2.1
        assert isinstance(frame, np.ndarray)
        assert frame.shape == (120, 160, 3)
        assert frame.dtype == np.uint8


def test_framereader_render_mode(sample_clip: Path):
    """Render mode yields exact frame count == round(duration*fps)."""
    # 2s clip at 30 fps, full 320x240
    reader = FrameReader(sample_clip, out_w=320, out_h=240, fps=30, mode="render")
    frames = list(reader)

    # Render frame count == round(duration * fps) = round(2.0 * 30) = 60
    assert len(frames) == 60

    for idx, (index, t, frame) in enumerate(frames):
        assert index == idx
        assert frame.shape == (240, 320, 3)
        assert frame.dtype == np.uint8


def test_framereader_segments_respected(sample_clip: Path):
    """start_s and end_s segments are respected in analysis and render modes."""
    # Segment [0.5, 1.5] -> duration 1.0s
    # In analysis mode: 10 fps -> ~10 frames
    with FrameReader(sample_clip, out_w=160, out_h=120, fps=10, start_s=0.5, end_s=1.5, mode="analysis") as reader:
        frames_analysis = list(reader)
    assert 9 <= len(frames_analysis) <= 11
    for index, t, frame in frames_analysis:
        assert 0.5 <= t <= 1.5
        assert frame.shape == (120, 160, 3)

    # In render mode: 30 fps -> round((1.5 - 0.5) * 30) = 30 frames
    with FrameReader(sample_clip, out_w=320, out_h=240, fps=30, start_s=0.5, end_s=1.5, mode="render") as reader:
        frames_render = list(reader)
    assert len(frames_render) == 30
    for index, t, frame in frames_render:
        assert 0.5 <= t <= 1.55
        assert frame.shape == (240, 320, 3)
