"""Media probing and VideoInfo metadata per CONTRACT §5."""

import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Optional, Union
from pydantic import BaseModel, ConfigDict, Field

from reframe.config import Config, load_config
from reframe.errors import InvalidInputError
from reframe.ffmpeg_utils import require_tools


class VideoInfo(BaseModel):
    """Video metadata model per CONTRACT §5."""

    model_config = ConfigDict(extra="forbid")

    path: str
    sha256: str
    size_bytes: int
    width: int
    height: int
    fps: float
    is_vfr: bool
    duration_s: float
    rotation: int
    video_codec: str
    has_audio: bool
    audio_codec: Optional[str] = None
    n_frames_expected: int
    warnings: list[str] = Field(default_factory=list)


def _parse_fraction(frac: str) -> float:
    """Parse 'num/den' string from ffprobe into float."""
    if not frac or "/" not in frac:
        return 0.0
    parts = frac.split("/", 1)
    try:
        num = float(parts[0])
        den = float(parts[1])
        return num / den if den != 0.0 else 0.0
    except (ValueError, TypeError):
        return 0.0


def probe(
    path: Union[Path, str],
    cfg: Optional[Config] = None,
) -> VideoInfo:
    """Probe video file and return VideoInfo per CONTRACT §5."""
    require_tools()
    p = Path(path)

    if not p.exists():
        raise InvalidInputError(f"File not found: {path}", code="file_not_found")
    if p.is_dir():
        raise InvalidInputError(f"Path is a directory, not a media file: {path}", code="not_media")

    size_bytes = p.stat().st_size
    if size_bytes == 0:
        raise InvalidInputError(f"File is empty (0 bytes): {path}", code="empty_file")

    # Compute sha256 streamed in 1 MB chunks
    hasher = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(1024 * 1024):
            hasher.update(chunk)
    sha256 = hasher.hexdigest()

    # ffprobe -v error -show_streams -show_format -of json
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_streams",
        "-show_format",
        "-of",
        "json",
        p.as_posix(),
    ]

    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
    except subprocess.TimeoutExpired as exc:
        raise InvalidInputError(f"ffprobe timed out on {path}", code="not_media") from exc

    if res.returncode != 0 or not res.stdout.strip():
        raise InvalidInputError(f"File is not valid media: {path}", code="not_media")

    try:
        data: dict[str, Any] = json.loads(res.stdout)
    except json.JSONDecodeError as exc:
        raise InvalidInputError(f"Failed to parse ffprobe json output for {path}", code="not_media") from exc

    streams = data.get("streams", [])
    format_info = data.get("format", {})

    video_stream = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video_stream is None:
        raise InvalidInputError(f"No video stream found in {path}", code="no_video_stream")

    audio_stream = next((s for s in streams if s.get("codec_type") == "audio"), None)
    has_audio = audio_stream is not None
    audio_codec = audio_stream.get("codec_name") if has_audio else None

    # Parse duration
    duration_str = format_info.get("duration") or video_stream.get("duration")
    if not duration_str or duration_str in ("N/A", "none", "None"):
        raise InvalidInputError(f"Missing duration in {path}", code="bad_duration")

    try:
        duration_s = float(duration_str)
    except (ValueError, TypeError) as exc:
        raise InvalidInputError(f"Invalid duration '{duration_str}' in {path}", code="bad_duration") from exc

    if duration_s <= 0.0:
        raise InvalidInputError(f"Non-positive duration {duration_s} in {path}", code="bad_duration")

    duration_s = round(duration_s, 3)

    if cfg is None:
        cfg = load_config()

    if duration_s > cfg.runtime.max_duration_s:
        raise InvalidInputError(
            f"Duration {duration_s}s exceeds max_duration_s {cfg.runtime.max_duration_s}s",
            code="too_long",
        )

    # Dimensions
    try:
        width = int(video_stream.get("width", 0))
        height = int(video_stream.get("height", 0))
    except (ValueError, TypeError):
        width, height = 0, 0

    if width <= 0 or height <= 0:
        raise InvalidInputError(f"Invalid video dimensions {width}x{height} in {path}", code="not_media")

    # Rotation from side_data_list or tags
    rotation = 0
    for sd in video_stream.get("side_data_list", []):
        if "rotation" in sd:
            try:
                rotation = int(float(sd["rotation"]))
                break
            except (ValueError, TypeError):
                pass

    if rotation == 0:
        for k, v in video_stream.get("tags", {}).items():
            if k.lower() == "rotate":
                try:
                    rotation = int(float(v))
                    break
                except (ValueError, TypeError):
                    pass

    if rotation == 0:
        for k, v in format_info.get("tags", {}).items():
            if k.lower() == "rotate":
                try:
                    rotation = int(float(v))
                    break
                except (ValueError, TypeError):
                    pass

    rotation = int(rotation) % 360
    if rotation > 180:
        rotation -= 360

    # Swap width/height when rotation is ±90
    if abs(rotation) in (90, 270):
        width, height = height, width

    # Framerate & VFR
    r_frame_rate = video_stream.get("r_frame_rate", "0/0")
    avg_frame_rate = video_stream.get("avg_frame_rate", "0/0")

    r_fps = _parse_fraction(r_frame_rate)
    avg_fps = _parse_fraction(avg_frame_rate)
    fps = avg_fps if avg_fps > 0.0 else r_fps

    if fps <= 0.0:
        raise InvalidInputError(f"Invalid frame rate for {path}", code="bad_duration")

    is_vfr = (r_frame_rate != avg_frame_rate) and avg_fps > 0.0 and r_fps > 0.0
    n_frames_expected = round(duration_s * fps)

    warnings: list[str] = []
    if not has_audio:
        warnings.append("no_audio")
    if is_vfr:
        warnings.append("vfr_normalised_to_cfr")

    return VideoInfo(
        path=p.as_posix(),
        sha256=sha256,
        size_bytes=size_bytes,
        width=width,
        height=height,
        fps=round(fps, 3),
        is_vfr=is_vfr,
        duration_s=duration_s,
        rotation=rotation,
        video_codec=video_stream.get("codec_name", "unknown"),
        has_audio=has_audio,
        audio_codec=audio_codec,
        n_frames_expected=n_frames_expected,
        warnings=warnings,
    )


__all__ = ["VideoInfo", "probe"]
