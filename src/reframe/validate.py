"""Output validation per CONTRACT §12."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Optional, Union

from reframe.errors import InvalidInputError, OutputValidationError
from reframe.ffmpeg_utils import require_tools
from reframe.timeline import load_timeline


def _ffprobe_output_info(path: Path) -> dict[str, Any]:
    """Probe an output MP4 for the fields validate_output needs."""
    require_tools()
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_streams",
        "-show_format",
        "-of",
        "json",
        path.as_posix(),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if res.returncode != 0 or not res.stdout.strip():
        raise OutputValidationError(
            f"ffprobe failed to read output {path}: {res.stderr.strip()}", code="unreadable_output"
        )
    data = json.loads(res.stdout)
    streams = data.get("streams", [])
    format_info = data.get("format", {})

    video_stream = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio_stream = next((s for s in streams if s.get("codec_type") == "audio"), None)

    width = int(video_stream["width"]) if video_stream else 0
    height = int(video_stream["height"]) if video_stream else 0

    duration_str = format_info.get("duration") or (video_stream or {}).get("duration")
    try:
        duration_s = float(duration_str) if duration_str else 0.0
    except (ValueError, TypeError):
        duration_s = 0.0

    return {
        "width": width,
        "height": height,
        "duration_s": duration_s,
        "has_audio": audio_stream is not None,
        "audio_codec": audio_stream.get("codec_name") if audio_stream else None,
    }


def _audio_md5(path: Path) -> Optional[str]:
    """Decoded-audio MD5 via `ffmpeg -i X -map 0:a -f md5 -`; None if no audio / on failure."""
    require_tools()
    cmd = ["ffmpeg", "-v", "error", "-i", path.as_posix(), "-map", "0:a", "-f", "md5", "-"]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except subprocess.TimeoutExpired:
        return None
    if res.returncode != 0:
        return None
    return res.stdout.strip() or None


def validate_output(run_dir: Union[Path, str]) -> dict:
    """Validate RUN_DIR's rendered output per CONTRACT §12.

    Always writes RUN_DIR/validation.json ({ok, checks{name:{ok, detail}}}), then
    raises OutputValidationError (exit 5) if any check failed.
    """
    run_dir = Path(run_dir)
    timeline_path = run_dir / "decision_timeline.json"
    if not timeline_path.is_file():
        raise InvalidInputError(
            f"decision_timeline.json not found in {run_dir}", code="file_not_found"
        )
    timeline = load_timeline(timeline_path)

    output_path = run_dir / "output.mp4"
    checks: dict[str, dict] = {}

    output_exists = output_path.is_file()
    checks["output_exists"] = {"ok": output_exists, "detail": output_path.as_posix()}

    crop_w, crop_h = timeline.target.crop_w, timeline.target.crop_h
    fps = timeline.target.fps
    expected_duration = timeline.segment.end_s - timeline.segment.start_s
    tol = max(2.0 / fps, 0.1) if fps > 0 else 0.1

    if output_exists:
        info = _ffprobe_output_info(output_path)

        dims_ok = (info["width"], info["height"]) == (crop_w, crop_h)
        checks["dims"] = {
            "ok": dims_ok,
            "detail": f"expected {crop_w}x{crop_h}, got {info['width']}x{info['height']}",
        }

        dur_diff = abs(info["duration_s"] - expected_duration)
        dur_ok = dur_diff <= tol
        checks["duration"] = {
            "ok": dur_ok,
            "detail": (
                f"expected {expected_duration:.3f}s (+/- {tol:.3f}s), got {info['duration_s']:.3f}s"
            ),
        }

        audio_ok = info["has_audio"] == timeline.input.has_audio
        checks["audio_present"] = {
            "ok": audio_ok,
            "detail": f"expected has_audio={timeline.input.has_audio}, got {info['has_audio']}",
        }

        is_full_clip = (
            timeline.segment.start_s <= 1e-6
            and expected_duration >= timeline.input.duration_s - 1e-3
        )
        audio_copied = (
            timeline.input.has_audio
            and info["has_audio"]
            and info["audio_codec"] == timeline.input.audio_codec
        )
        if is_full_clip and audio_copied:
            md5_in = _audio_md5(Path(timeline.input.path))
            md5_out = _audio_md5(output_path)
            md5_ok = md5_in is not None and md5_in == md5_out
            checks["audio_md5_match"] = {
                "ok": md5_ok,
                "detail": f"input={md5_in} output={md5_out}",
            }
        else:
            checks["audio_md5_match"] = {
                "ok": True,
                "detail": "skipped (not a full-clip run with copied audio)",
            }
    else:
        for name in ("dims", "duration", "audio_present", "audio_md5_match"):
            checks[name] = {"ok": False, "detail": "output.mp4 not found"}

    violations_ok = timeline.invariant_violations.count == 0
    checks["invariant_violations"] = {
        "ok": violations_ok,
        "detail": f"count={timeline.invariant_violations.count}",
    }

    ok = all(c["ok"] for c in checks.values())
    result = {"ok": ok, "checks": checks}

    validation_path = run_dir / "validation.json"
    with open(validation_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1)

    if not ok:
        failed = ", ".join(name for name, c in checks.items() if not c["ok"])
        raise OutputValidationError(
            f"Output validation failed for {run_dir}: {failed}", code="validation_failed"
        )

    return result


__all__ = ["validate_output"]
