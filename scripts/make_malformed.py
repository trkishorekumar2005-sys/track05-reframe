"""Generate malformed media files for test and validation per CONTRACT §5."""

import argparse
from pathlib import Path
import subprocess
import sys


def make_malformed(source: Path, out_dir: Path) -> None:
    """Generate 5 deterministic malformed files:

    - zero.mp4 (0 bytes)
    - text.mp4 (plain text)
    - truncated.mp4 (first 20% of bytes)
    - audio_only.mp4 (ffmpeg -vn -c:a copy)
    - no_audio.mp4 (ffmpeg -an -c:v copy)
    """
    if not source.is_file():
        raise FileNotFoundError(f"Source file not found: {source}")

    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. zero.mp4 (0 bytes)
    zero_file = out_dir / "zero.mp4"
    zero_file.write_bytes(b"")

    # 2. text.mp4 (plain text)
    text_file = out_dir / "text.mp4"
    text_file.write_text("This is plain text, not a valid media file.\n", encoding="utf-8")

    # 3. truncated.mp4 (first 20% of bytes)
    source_bytes = source.read_bytes()
    cutoff = int(len(source_bytes) * 0.2)
    truncated_file = out_dir / "truncated.mp4"
    truncated_file.write_bytes(source_bytes[:cutoff])

    # 4. audio_only.mp4 (ffmpeg -vn -c:a copy)
    audio_only_file = out_dir / "audio_only.mp4"
    cmd_audio = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-i",
        source.as_posix(),
        "-vn",
        "-c:a",
        "copy",
        audio_only_file.as_posix(),
    ]
    subprocess.run(cmd_audio, check=True, timeout=30)

    # 5. no_audio.mp4 (ffmpeg -an -c:v copy)
    no_audio_file = out_dir / "no_audio.mp4"
    cmd_no_audio = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-i",
        source.as_posix(),
        "-an",
        "-c:v",
        "copy",
        no_audio_file.as_posix(),
    ]
    subprocess.run(cmd_no_audio, check=True, timeout=30)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate malformed test clips.")
    parser.add_argument("--source", type=Path, required=True, help="Source MP4 clip")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/malformed"),
        help="Output directory (default: data/malformed)",
    )
    args = parser.parse_args()

    try:
        make_malformed(args.source, args.out)
        print(f"Malformed clips written to {args.out}")
    except Exception as exc:
        print(f"Error generating malformed clips: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
