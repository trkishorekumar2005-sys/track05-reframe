"""End-to-end CLI tests per CONTRACT §16."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from reframe.probe import probe

FIXTURE = Path(__file__).resolve().parent.parent / "fixtures" / "single_small.mp4"


def _run_cli(args: list[str], timeout: int = 300) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "reframe", *args],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def test_run_full_pipeline_9x16(tmp_path: Path):
    out_dir = tmp_path / "run9x16"
    res = _run_cli(["run", str(FIXTURE), "--aspect", "9:16", "--out", str(out_dir)])
    if res.returncode == 3:
        pytest.skip(
            "Model files missing/invalid; run `uv run python scripts/download_models.py` first."
        )
    assert res.returncode == 0, res.stderr

    validation = json.loads((out_dir / "validation.json").read_text(encoding="utf-8"))
    assert validation["ok"] is True


def test_run_zero_byte_input_exits_2(tmp_path: Path):
    empty = tmp_path / "empty.mp4"
    empty.write_bytes(b"")
    out_dir = tmp_path / "run_empty"
    res = _run_cli(["run", str(empty), "--aspect", "9:16", "--out", str(out_dir)], timeout=30)
    assert res.returncode == 2
    assert "ERROR" in res.stderr


def test_run_aspect_1x1_square_output(tmp_path: Path):
    out_dir = tmp_path / "run1x1"
    res = _run_cli(["run", str(FIXTURE), "--aspect", "1:1", "--out", str(out_dir)])
    if res.returncode == 3:
        pytest.skip(
            "Model files missing/invalid; run `uv run python scripts/download_models.py` first."
        )
    assert res.returncode == 0, res.stderr

    info = probe(out_dir / "output.mp4")
    assert info.width == info.height
