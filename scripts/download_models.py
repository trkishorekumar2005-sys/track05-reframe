"""Download model files and optionally write/verify manifest.lock.json.

This is the ONLY file in the project that makes network requests.
Runtime src/ code is fully offline.
"""

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MANIFEST_PATH = MODELS_DIR / "manifest.json"
LOCK_PATH = MODELS_DIR / "manifest.lock.json"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def _load_manifest() -> dict:
    if not MANIFEST_PATH.is_file():
        print(f"ERROR: manifest.json not found at {MANIFEST_PATH}", file=sys.stderr)
        sys.exit(1)
    with open(MANIFEST_PATH, "r") as f:
        return json.load(f)


def download_missing(manifest: dict) -> None:
    """Download any model files not already present in models/."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for name, entry in manifest.items():
        filename = entry["filename"]
        url = entry["url"]
        dest = MODELS_DIR / filename
        if dest.is_file():
            print(f"  [skip] {filename} already present")
            continue
        print(f"  [download] {filename} from {url}")
        try:
            urllib.request.urlretrieve(url, dest)
            print(f"  [done]  {filename} ({dest.stat().st_size:,} bytes)")
        except Exception as exc:
            print(f"  [error] Failed to download {filename}: {exc}", file=sys.stderr)
            sys.exit(1)


def write_lock(manifest: dict) -> None:
    """Compute sha256 + size for each model and write manifest.lock.json."""
    lock: dict = {}
    for name, entry in manifest.items():
        filename = entry["filename"]
        path = MODELS_DIR / filename
        if not path.is_file():
            print(f"ERROR: {filename} not found. Run without --lock first to download.", file=sys.stderr)
            sys.exit(1)
        sha = _sha256_file(path)
        size = path.stat().st_size
        lock[name] = {
            "filename": filename,
            "sha256": sha,
            "size_bytes": size,
            "size_mb": round(size / (1024 * 1024), 3),
        }
        print(f"  [lock]  {filename}: sha256={sha[:16]}... size={size:,} bytes")

    with open(LOCK_PATH, "w") as f:
        json.dump(lock, f, indent=2)
    print(f"\nLock file written: {LOCK_PATH}")


def verify_against_lock(manifest: dict) -> None:
    """Verify each model file SHA256 matches the lock file; exit non-zero on mismatch."""
    if not LOCK_PATH.is_file():
        print(
            f"ERROR: manifest.lock.json not found at {LOCK_PATH}.\n"
            "Run `uv run python scripts/download_models.py --lock` to create it.",
            file=sys.stderr,
        )
        sys.exit(1)

    with open(LOCK_PATH, "r") as f:
        lock = json.load(f)

    failures: list[str] = []
    for name, entry in manifest.items():
        filename = entry["filename"]
        path = MODELS_DIR / filename

        if name not in lock:
            failures.append(f"  {filename}: missing from lock file")
            continue

        expected_sha = lock[name]["sha256"]

        if not path.is_file():
            failures.append(f"  {filename}: file not found (expected sha256={expected_sha[:16]}...)")
            continue

        actual_sha = _sha256_file(path)
        if actual_sha != expected_sha:
            failures.append(
                f"  {filename}: sha256 mismatch\n"
                f"    expected: {expected_sha}\n"
                f"    actual:   {actual_sha}"
            )
        else:
            print(f"  [ok]    {filename}: sha256 matches")

    if failures:
        print("\nVerification FAILED:", file=sys.stderr)
        for msg in failures:
            print(msg, file=sys.stderr)
        print(
            "\nRun `uv run python scripts/download_models.py --lock` to regenerate the lock file.",
            file=sys.stderr,
        )
        sys.exit(1)
    else:
        print("\nAll models verified OK.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and verify reframe model files.")
    parser.add_argument(
        "--lock",
        action="store_true",
        help="Download missing models then compute sha256 and write manifest.lock.json",
    )
    args = parser.parse_args()

    manifest = _load_manifest()

    if args.lock:
        download_missing(manifest)
        write_lock(manifest)
    else:
        verify_against_lock(manifest)


if __name__ == "__main__":
    main()
