"""Logging configuration for reframe."""

import datetime
import json
import logging
import sys
from pathlib import Path
from typing import Any, Union


class JsonLinesFormatter(logging.Formatter):
    """Formats LogRecords into JSON Lines with ts, level, stage, event, msg + extras."""

    STANDARD_ATTRS = {
        "args",
        "asctime",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "module",
        "msecs",
        "message",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "thread",
        "threadName",
        "taskName",
        "stage",
        "event",
        "ts",
    }

    def format(self, record: logging.LogRecord) -> str:
        data: dict[str, Any] = {
            "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "level": record.levelname,
            "stage": getattr(record, "stage", None),
            "event": getattr(record, "event", None),
            "msg": record.getMessage(),
        }
        for k, v in record.__dict__.items():
            if k not in self.STANDARD_ATTRS:
                data[k] = v
        if record.exc_info:
            data["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(data)


def setup_logging(
    run_dir: Union[Path, str, None] = None,
    verbose: bool = False,
) -> logging.Logger:
    """Setup console and optional JSON lines logging per CONTRACT §2."""
    logger = logging.getLogger("reframe")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)

    # Reset any existing handlers to allow safe reconfiguration
    logger.handlers.clear()

    # Console handler (human-readable)
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    console_fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    console_handler.setFormatter(console_fmt)
    logger.addHandler(console_handler)

    # File handler (JSON lines to run_dir/logs/run.jsonl)
    if run_dir is not None:
        logs_dir = Path(run_dir) / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        jsonl_path = logs_dir / "run.jsonl"
        file_handler = logging.FileHandler(jsonl_path, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
        file_handler.setFormatter(JsonLinesFormatter())
        logger.addHandler(file_handler)

    # Avoid propagation to root logger to prevent duplicate messages
    logger.propagate = False
    return logger


__all__ = ["setup_logging", "JsonLinesFormatter"]
