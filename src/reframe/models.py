"""Model file resolution and integrity check per CONTRACT §2."""

import hashlib
import json
from pathlib import Path
from typing import Optional

from reframe.errors import MissingDependencyError

# Resolved relative to this file: src/reframe/models.py → repo root
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_MODELS_DIR = _REPO_ROOT / "models"
_LOCK_PATH = _MODELS_DIR / "manifest.lock.json"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def _load_lock() -> dict:
    """Load manifest.lock.json or raise MissingDependencyError."""
    if not _LOCK_PATH.is_file():
        raise MissingDependencyError(
            f"models/manifest.lock.json not found. "
            f"Run: uv run python scripts/download_models.py --lock",
            code="ffmpeg_missing",  # reuse 'ffmpeg_missing' bucket → exit 3
        )
    with open(_LOCK_PATH, "r") as f:
        return json.load(f)


def ensure_model(name: str, lock: Optional[dict] = None) -> Path:
    """Verify model *name* exists and matches its sha256; return its Path.

    Raises MissingDependencyError (exit 3) if the file is absent or corrupted,
    instructing the user to run ``uv run python scripts/download_models.py``.
    """
    if lock is None:
        lock = _load_lock()

    if name not in lock:
        raise MissingDependencyError(
            f"Model '{name}' not found in manifest.lock.json. "
            f"Run: uv run python scripts/download_models.py --lock",
            code="ffmpeg_missing",
        )

    entry = lock[name]
    filename = entry["filename"]
    expected_sha = entry["sha256"]
    model_path = _MODELS_DIR / filename

    if not model_path.is_file():
        raise MissingDependencyError(
            f"Model file '{filename}' not found in models/. "
            f"Run: uv run python scripts/download_models.py",
            code="ffmpeg_missing",
        )

    actual_sha = _sha256_file(model_path)
    if actual_sha != expected_sha:
        raise MissingDependencyError(
            f"Model '{filename}' sha256 mismatch (expected {expected_sha[:16]}..., "
            f"got {actual_sha[:16]}...). "
            f"Run: uv run python scripts/download_models.py --lock",
            code="ffmpeg_missing",
        )

    return model_path


__all__ = ["ensure_model"]
