"""Unit tests for logging configuration per CONTRACT §2."""

import json
from pathlib import Path

from reframe.log import setup_logging


def test_setup_logging_jsonl(tmp_path: Path):
    """Logging writes JSON Lines to run_dir/logs/run.jsonl with required keys."""
    logger = setup_logging(run_dir=tmp_path, verbose=True)
    logger.info(
        "Starting test event",
        extra={"stage": "analysis", "event": "start", "extra_metric": 42},
    )

    log_file = tmp_path / "logs" / "run.jsonl"
    assert log_file.is_file()

    lines = log_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])

    assert "ts" in record
    assert record["level"] == "INFO"
    assert record["stage"] == "analysis"
    assert record["event"] == "start"
    assert record["msg"] == "Starting test event"
    assert record["extra_metric"] == 42
