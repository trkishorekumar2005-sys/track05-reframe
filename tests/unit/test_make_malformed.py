"""Unit tests for scripts/make_malformed.py."""

from pathlib import Path
import subprocess
import sys
import pytest


@pytest.fixture
def source_clip(tmp_path: Path) -> Path:
    """Generate a test clip with both audio and video."""
    clip = tmp_path / "source.mp4"
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "testsrc=duration=1:size=320x240:rate=30",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=1000:duration=1",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        clip.as_posix(),
    ]
    subprocess.run(cmd, check=True, capture_output=True, timeout=15)
    return clip


def test_make_malformed_script(source_clip: Path, tmp_path: Path):
    """Running scripts/make_malformed.py generates all 5 files deterministically."""
    out_dir = tmp_path / "malformed"
    cmd = [
        sys.executable,
        "scripts/make_malformed.py",
        "--source",
        source_clip.as_posix(),
        "--out",
        out_dir.as_posix(),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
    assert result.returncode == 0

    zero_file = out_dir / "zero.mp4"
    text_file = out_dir / "text.mp4"
    trunc_file = out_dir / "truncated.mp4"
    audio_only_file = out_dir / "audio_only.mp4"
    no_audio_file = out_dir / "no_audio.mp4"

    assert zero_file.is_file()
    assert zero_file.stat().st_size == 0

    assert text_file.is_file()
    assert "plain text" in text_file.read_text(encoding="utf-8")

    assert trunc_file.is_file()
    source_len = source_clip.stat().st_size
    assert trunc_file.stat().st_size == int(source_len * 0.2)

    assert audio_only_file.is_file()
    assert audio_only_file.stat().st_size > 0

    assert no_audio_file.is_file()
    assert no_audio_file.stat().st_size > 0
