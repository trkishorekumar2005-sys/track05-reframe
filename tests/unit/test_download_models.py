"""Unit tests for scripts/download_models.py lock verification (no network)."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

_SCRIPT_PATH = Path(__file__).resolve().parent.parent.parent / "scripts" / "download_models.py"


def _load_module(tmp_path: Path):
    """Import download_models.py fresh, redirecting its paths into tmp_path."""
    spec = importlib.util.spec_from_file_location("download_models_test", _SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.MODELS_DIR = tmp_path
    module.MANIFEST_PATH = tmp_path / "manifest.json"
    module.LOCK_PATH = tmp_path / "manifest.lock.json"
    return module


def test_verify_ok(tmp_path: Path):
    """verify_against_lock() passes silently when sha256 matches."""
    module = _load_module(tmp_path)
    content = b"fake model bytes"
    (tmp_path / "model.bin").write_bytes(content)
    manifest = {"model": {"filename": "model.bin", "url": "https://example.invalid/model.bin"}}
    lock = {
        "model": {
            "filename": "model.bin",
            "sha256": hashlib.sha256(content).hexdigest(),
            "size_bytes": len(content),
            "size_mb": round(len(content) / (1024 * 1024), 3),
        }
    }
    module.LOCK_PATH.write_text(json.dumps(lock), encoding="utf-8")

    module.verify_against_lock(manifest)  # must not raise/exit


def test_verify_mismatch_exits_nonzero(tmp_path: Path):
    """verify_against_lock() exits non-zero on sha256 mismatch."""
    module = _load_module(tmp_path)
    (tmp_path / "model.bin").write_bytes(b"corrupted content")
    manifest = {"model": {"filename": "model.bin", "url": "https://example.invalid/model.bin"}}
    lock = {"model": {"filename": "model.bin", "sha256": "0" * 64, "size_bytes": 5, "size_mb": 0.0}}
    module.LOCK_PATH.write_text(json.dumps(lock), encoding="utf-8")

    with pytest.raises(SystemExit) as exc_info:
        module.verify_against_lock(manifest)
    assert exc_info.value.code != 0


def test_verify_missing_model_file_exits_nonzero(tmp_path: Path):
    """verify_against_lock() exits non-zero when a locked model file is absent."""
    module = _load_module(tmp_path)
    manifest = {"model": {"filename": "model.bin", "url": "https://example.invalid/model.bin"}}
    lock = {"model": {"filename": "model.bin", "sha256": "0" * 64, "size_bytes": 5, "size_mb": 0.0}}
    module.LOCK_PATH.write_text(json.dumps(lock), encoding="utf-8")

    with pytest.raises(SystemExit) as exc_info:
        module.verify_against_lock(manifest)
    assert exc_info.value.code != 0


def test_verify_missing_lock_file_exits_nonzero(tmp_path: Path):
    """verify_against_lock() exits non-zero if manifest.lock.json is absent."""
    module = _load_module(tmp_path)
    manifest = {"model": {"filename": "model.bin", "url": "https://example.invalid/model.bin"}}

    with pytest.raises(SystemExit) as exc_info:
        module.verify_against_lock(manifest)
    assert exc_info.value.code != 0


def test_write_lock_computes_sha256_and_size(tmp_path: Path):
    """write_lock() computes sha256 + size and writes manifest.lock.json (no network)."""
    module = _load_module(tmp_path)
    content = b"another fake model payload"
    (tmp_path / "model.bin").write_bytes(content)
    manifest = {"model": {"filename": "model.bin", "url": "https://example.invalid/model.bin"}}

    module.write_lock(manifest)

    lock = json.loads(module.LOCK_PATH.read_text(encoding="utf-8"))
    assert lock["model"]["sha256"] == hashlib.sha256(content).hexdigest()
    assert lock["model"]["size_bytes"] == len(content)
