"""Unit tests for media probing, VideoInfo, and error codes per CONTRACT §5."""

from pathlib import Path
import subprocess
from unittest.mock import patch
import pytest

from reframe.config import Config, RuntimeConfig
from reframe.errors import InvalidInputError, MissingDependencyError
from reframe.probe import VideoInfo, probe


@pytest.fixture
def sample_video(tmp_path: Path) -> Path:
    """Generate a 2.0s 320x240 30fps test video with audio using lavfi."""
    out_file = tmp_path / "sample.mp4"
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "testsrc=duration=2:size=320x240:rate=30",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=1000:duration=2",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        out_file.as_posix(),
    ]
    subprocess.run(cmd, check=True, capture_output=True, timeout=15)
    return out_file


def test_probe_valid_media(sample_video: Path):
    """probe() correctly parses a valid MP4 file and streams sha256."""
    info = probe(sample_video)
    assert isinstance(info, VideoInfo)
    assert info.width == 320
    assert info.height == 240
    assert abs(info.fps - 30.0) < 0.1
    assert abs(info.duration_s - 2.0) < 0.1
    assert info.has_audio is True
    assert info.audio_codec == "aac"
    assert info.video_codec == "h264"
    assert info.n_frames_expected == round(info.duration_s * info.fps)
    assert len(info.sha256) == 64
    assert info.size_bytes > 0


def test_error_file_not_found(tmp_path: Path):
    """Non-existent file raises InvalidInputError with code='file_not_found'."""
    missing = tmp_path / "does_not_exist.mp4"
    with pytest.raises(InvalidInputError) as exc_info:
        probe(missing)
    assert exc_info.value.code == "file_not_found"
    assert exc_info.value.exit_code == 2


def test_error_empty_file(tmp_path: Path):
    """0-byte file raises InvalidInputError with code='empty_file'."""
    empty = tmp_path / "empty.mp4"
    empty.write_bytes(b"")
    with pytest.raises(InvalidInputError) as exc_info:
        probe(empty)
    assert exc_info.value.code == "empty_file"
    assert exc_info.value.exit_code == 2


def test_error_not_media(tmp_path: Path):
    """Plain text or corrupt file raises InvalidInputError with code='not_media'."""
    text_file = tmp_path / "test.txt"
    text_file.write_text("Not video content", encoding="utf-8")
    with pytest.raises(InvalidInputError) as exc_info:
        probe(text_file)
    assert exc_info.value.code == "not_media"
    assert exc_info.value.exit_code == 2


def test_error_no_video_stream(tmp_path: Path):
    """Audio-only MP4 raises InvalidInputError with code='no_video_stream'."""
    audio_file = tmp_path / "audio_only.mp4"
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=1000:duration=1",
        "-c:a",
        "aac",
        audio_file.as_posix(),
    ]
    subprocess.run(cmd, check=True, capture_output=True, timeout=15)
    with pytest.raises(InvalidInputError) as exc_info:
        probe(audio_file)
    assert exc_info.value.code == "no_video_stream"
    assert exc_info.value.exit_code == 2


def test_error_bad_duration(tmp_path: Path, sample_video: Path):
    """Media with invalid/missing duration raises InvalidInputError with code='bad_duration'."""
    # Mock ffprobe returning duration "0.0" or empty
    with patch("subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = """{
            "streams": [{"codec_type": "video", "width": 320, "height": 240, "r_frame_rate": "30/1"}],
            "format": {"duration": "0.0"}
        }"""
        with pytest.raises(InvalidInputError) as exc_info:
            probe(sample_video)
        assert exc_info.value.code == "bad_duration"
        assert exc_info.value.exit_code == 2


def test_error_too_long(sample_video: Path):
    """Video exceeding runtime.max_duration_s raises InvalidInputError with code='too_long'."""
    cfg = Config(runtime=RuntimeConfig(max_duration_s=1.0))
    with pytest.raises(InvalidInputError) as exc_info:
        probe(sample_video, cfg=cfg)
    assert exc_info.value.code == "too_long"
    assert exc_info.value.exit_code == 2


def test_error_ffmpeg_missing(sample_video: Path):
    """Missing ffmpeg or ffprobe raises MissingDependencyError with code='ffmpeg_missing'."""
    with patch("shutil.which", return_value=None):
        with pytest.raises(MissingDependencyError) as exc_info:
            probe(sample_video)
        assert exc_info.value.code == "ffmpeg_missing"
        assert exc_info.value.exit_code == 3


def test_warning_no_audio(tmp_path: Path):
    """Video without audio passes with has_audio=False and warning 'no_audio'."""
    no_audio_file = tmp_path / "no_audio.mp4"
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "testsrc=duration=1:size=320x240:rate=30",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-an",
        no_audio_file.as_posix(),
    ]
    subprocess.run(cmd, check=True, capture_output=True, timeout=15)
    info = probe(no_audio_file)
    assert info.has_audio is False
    assert info.audio_codec is None
    assert "no_audio" in info.warnings


def test_warning_vfr(sample_video: Path):
    """VFR video (r_frame_rate != avg_frame_rate) adds warning 'vfr_normalised_to_cfr'."""
    with patch("subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = """{
            "streams": [{
                "codec_type": "video",
                "width": 320,
                "height": 240,
                "r_frame_rate": "30/1",
                "avg_frame_rate": "2997/100"
            }],
            "format": {"duration": "2.0"}
        }"""
        info = probe(sample_video)
        assert info.is_vfr is True
        assert "vfr_normalised_to_cfr" in info.warnings


def test_rotation_swaps_dimensions(tmp_path: Path):
    """Rotation of ±90 degrees swaps width and height per CONTRACT §5."""
    # Test via MKV tag (supported by Matroska container)
    rot_file = tmp_path / "rot90.mkv"
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "testsrc=duration=1:size=320x240:rate=30",
        "-metadata:s:v:0",
        "rotate=90",
        "-c:v",
        "libx264",
        rot_file.as_posix(),
    ]
    subprocess.run(cmd, check=True, capture_output=True, timeout=15)
    info = probe(rot_file)
    assert info.rotation == 90
    assert info.width == 240
    assert info.height == 320

    # Test via side_data_list
    with patch("subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = """{
            "streams": [{
                "codec_type": "video",
                "width": 1920,
                "height": 1080,
                "r_frame_rate": "30/1",
                "side_data_list": [{"rotation": -90}]
            }],
            "format": {"duration": "1.0"}
        }"""
        side_info = probe(rot_file)
        assert side_info.rotation == -90
        assert side_info.width == 1080
        assert side_info.height == 1920


def test_cli_validate_command(sample_video: Path):
    """reframe validate prints VideoInfo as indented JSON."""
    import sys
    cmd = [sys.executable, "-m", "reframe", "validate", sample_video.as_posix()]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0
    import json
    data = json.loads(result.stdout)
    assert data["width"] == 320
    assert data["height"] == 240
    assert data["video_codec"] == "h264"
