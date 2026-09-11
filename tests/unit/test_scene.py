"""Unit tests for scene cut detection per CONTRACT §7."""

from pathlib import Path
import subprocess

import pytest

from reframe.config import Config, SceneConfig
from reframe.ffmpeg_utils import FrameReader
from reframe.scene import SceneAnalyzer


@pytest.fixture
def hard_cut_clip(tmp_path: Path) -> Path:
    """3s clip: 1.5s of `testsrc` then 1.5s of the visually distinct `testsrc2` pattern."""
    clip = tmp_path / "cut.mp4"
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "testsrc=duration=1.5:size=320x240:rate=10",
        "-f",
        "lavfi",
        "-i",
        "testsrc2=duration=1.5:size=320x240:rate=10",
        "-filter_complex",
        "[0:v][1:v]concat=n=2:v=1:a=0[v]",
        "-map",
        "[v]",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        clip.as_posix(),
    ]
    subprocess.run(cmd, check=True, capture_output=True, timeout=30)
    return clip


def test_hard_cut_between_different_sources_is_detected(hard_cut_clip: Path):
    # testsrc -> testsrc2 is a real but modest histogram jump (~0.39 Bhattacharyya
    # distance) against a same-content baseline around ~0.08-0.09; a threshold in
    # between isolates the cut robustly without depending on the exact spike size.
    cfg = Config(scene=SceneConfig(cut_threshold=0.3))
    analyzer = SceneAnalyzer(cfg)

    results = []
    with FrameReader(hard_cut_clip, out_w=320, out_h=240, fps=10, mode="analysis") as reader:
        for i, t, frame in reader:
            is_cut, _ = analyzer.analyze(frame, n_faces=0)
            results.append((i, t, is_cut))

    cut_indices = [i for i, _, is_cut in results if is_cut]
    assert cut_indices, "expected at least one detected cut"
    # 3s @ 10fps -> 30 frames; the concat boundary sits at index ~15.
    assert any(abs(i - 15) <= 2 for i in cut_indices)
    # Should be a rare event, not firing on most frames.
    assert len(cut_indices) < len(results) // 2


def test_no_cut_within_a_single_source():
    cfg = Config()
    analyzer = SceneAnalyzer(cfg)

    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        clip = Path(tmp) / "steady.mp4"
        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=duration=2:size=320x240:rate=10",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            clip.as_posix(),
        ]
        subprocess.run(cmd, check=True, capture_output=True, timeout=30)

        with FrameReader(clip, out_w=320, out_h=240, fps=10, mode="analysis") as reader:
            cuts = [analyzer.analyze(frame, n_faces=0)[0] for _, t, frame in reader]

    assert not any(cuts)
