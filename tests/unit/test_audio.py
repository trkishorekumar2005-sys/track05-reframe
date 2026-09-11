"""Unit tests for audio feature extraction per CONTRACT §7."""

from pathlib import Path
import subprocess

import pytest

from reframe.audio import compute_audio_features
from reframe.config import Config
from reframe.probe import probe


def _make_clip(tmp_path: Path, name: str, audio_args: list[str], duration: float = 3.0) -> Path:
    clip = tmp_path / name
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"testsrc=duration={duration}:size=320x240:rate=30",
        *audio_args,
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        clip.as_posix(),
    ]
    subprocess.run(cmd, check=True, capture_output=True, timeout=30)
    return clip


@pytest.fixture
def silence_clip(tmp_path: Path) -> Path:
    return _make_clip(
        tmp_path,
        "silence.mp4",
        ["-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono", "-c:a", "aac", "-shortest"],
    )


@pytest.fixture
def no_audio_clip(tmp_path: Path) -> Path:
    return _make_clip(tmp_path, "no_audio.mp4", ["-an"])


def _tick_times(start_s: float, end_s: float, sample_fps: float) -> list[float]:
    times = []
    k = 0
    while True:
        t = start_s + k / sample_fps
        if t >= end_s:
            break
        times.append(round(t, 3))
        k += 1
    return times


def test_silence_gives_near_zero_vad(tmp_path: Path, silence_clip: Path):
    cfg = Config()
    info = probe(silence_clip, cfg)
    tick_times = _tick_times(0.0, info.duration_s, cfg.analysis.sample_fps)

    features = compute_audio_features(info, cfg, tick_times, 0.0, info.duration_s, tmp_path)

    assert len(features.vad) == len(tick_times)
    assert max(features.vad) < 0.1
    assert features.vad_segments == []


def test_energy_length_matches_n_ticks(tmp_path: Path, silence_clip: Path):
    cfg = Config()
    info = probe(silence_clip, cfg)
    tick_times = _tick_times(0.0, info.duration_s, cfg.analysis.sample_fps)

    features = compute_audio_features(info, cfg, tick_times, 0.0, info.duration_s, tmp_path)

    assert len(features.energy) == len(tick_times)
    assert all(0.0 <= e <= 1.0 for e in features.energy)


def test_no_audio_gives_zeros_and_warning(tmp_path: Path, no_audio_clip: Path):
    cfg = Config()
    info = probe(no_audio_clip, cfg)
    assert info.has_audio is False
    tick_times = _tick_times(0.0, info.duration_s, cfg.analysis.sample_fps)

    features = compute_audio_features(info, cfg, tick_times, 0.0, info.duration_s, tmp_path)

    assert features.vad == [0.0] * len(tick_times)
    assert features.energy == [0.0] * len(tick_times)
    assert features.vad_segments == []
    assert "no_audio" in features.warnings
