"""Rendering per CONTRACT §11.

Decodes the ORIGINAL full-resolution video via FrameReader (render mode),
applies the per-frame crop (speaker) or blur_pad (fallback) recorded in the
timeline, and pipes raw frames to an ffmpeg encoder. Never re-runs analysis.
"""

from __future__ import annotations

import logging
import math
import subprocess
from pathlib import Path
from typing import Any, Optional, TextIO, Union

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter
from scipy.ndimage import zoom as ndi_zoom

from reframe.config import Config
from reframe.crop import check_crop_invariants
from reframe.errors import ProcessingError
from reframe.ffmpeg_utils import FrameReader, require_tools
from reframe.timeline import load_timeline
from reframe.track import box_at

logger = logging.getLogger("reframe")

_BLUR_KSIZE = 51


def _gaussian_sigma(ksize: int) -> float:
    """OpenCV-style kernel-size -> sigma mapping, so 'GaussianBlur 51' has a concrete meaning."""
    return 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8


def _resize(frame: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    """Bilinear-resize `frame` to exactly (out_h, out_w, 3)."""
    in_h, in_w = frame.shape[:2]
    if in_h == out_h and in_w == out_w:
        return frame

    zoom_factors = (out_h / in_h, out_w / in_w, 1.0)
    resized = ndi_zoom(frame, zoom_factors, order=1, mode="nearest")

    # scipy.ndimage.zoom can be off by a pixel due to float rounding; enforce exact shape.
    resized = resized[:out_h, :out_w]
    if resized.shape[0] != out_h or resized.shape[1] != out_w:
        padded = np.zeros((out_h, out_w, 3), dtype=frame.dtype)
        ph, pw = resized.shape[:2]
        padded[:ph, :pw] = resized
        resized = padded

    return np.clip(resized, 0, 255).astype(frame.dtype, copy=False)


def _blur(frame: np.ndarray, ksize: int = _BLUR_KSIZE) -> np.ndarray:
    sigma = _gaussian_sigma(ksize)
    blurred = gaussian_filter(frame.astype(np.float32), sigma=(sigma, sigma, 0.0))
    return np.clip(blurred, 0, 255).astype(np.uint8)


def _blur_pad(frame: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    """Fallback frame per CONTRACT §11: blurred cover background + centred fit foreground."""
    in_h, in_w = frame.shape[:2]

    cover_scale = max(out_w / in_w, out_h / in_h)
    cov_w = max(1, round(in_w * cover_scale))
    cov_h = max(1, round(in_h * cover_scale))
    bg = _resize(frame, cov_w, cov_h)
    x0 = max(0, (cov_w - out_w) // 2)
    y0 = max(0, (cov_h - out_h) // 2)
    bg = bg[y0 : y0 + out_h, x0 : x0 + out_w]
    if bg.shape[0] != out_h or bg.shape[1] != out_w:
        bg = _resize(np.ascontiguousarray(bg), out_w, out_h)
    bg = _blur(np.ascontiguousarray(bg))

    fit_scale = min(out_w / in_w, out_h / in_h)
    fit_w = max(1, round(in_w * fit_scale))
    fit_h = max(1, round(in_h * fit_scale))
    fg = _resize(frame, fit_w, fit_h)

    canvas = bg.copy()
    fx = (out_w - fit_w) // 2
    fy = (out_h - fit_h) // 2
    canvas[fy : fy + fit_h, fx : fx + fit_w] = fg
    return canvas


def _render_frame(frame: np.ndarray, crop: Any, crop_w: int, crop_h: int) -> np.ndarray:
    if crop.mode == "speaker":
        x, y, w, h = crop.x, crop.y, crop.w, crop.h
        out = frame[y : y + h, x : x + w]
        if out.shape[0] != h or out.shape[1] != w:
            out = _resize(np.ascontiguousarray(out), w, h)
        return np.ascontiguousarray(out)
    return _blur_pad(frame, crop_w, crop_h)


def _debug_dims(W: int, H: int, debug_width: int) -> tuple[int, int]:
    debug_w = min(int(debug_width), W)
    debug_w = max(2, debug_w - (debug_w % 2))
    scale = debug_w / W
    debug_h = int(round(H * scale))
    if debug_h % 2 != 0:
        debug_h += 1
    return debug_w, debug_h


def _draw_debug_frame(
    frame: np.ndarray,
    debug_w: int,
    debug_h: int,
    W: int,
    H: int,
    tracks_by_id: dict,
    t: float,
    decision: Any,
    crop: Any,
    in_speech: bool,
) -> np.ndarray:
    scale = debug_w / W
    small = _resize(frame, debug_w, debug_h)
    img = Image.fromarray(np.ascontiguousarray(small[:, :, ::-1]))  # BGR -> RGB
    draw = ImageDraw.Draw(img)

    for track_id, tr in tracks_by_id.items():
        box = box_at(tr, t)
        if box is None:
            continue
        x, y, w, h = (v * scale for v in box)
        score = decision.scores.get(track_id) if decision is not None else None
        label = f"{track_id}:{score:.2f}" if score is not None else str(track_id)
        draw.rectangle([x, y, x + w, y + h], outline=(0, 255, 0), width=2)
        draw.text((x, max(0.0, y - 12)), label, fill=(0, 255, 0))

    if crop.mode == "speaker":
        cx, cy, cw, ch = crop.x * scale, crop.y * scale, crop.w * scale, crop.h * scale
        draw.rectangle([cx, cy, cx + cw, cy + ch], outline=(255, 0, 0), width=3)

    if decision is not None:
        text = f"mode={decision.mode} reason={decision.reason} conf={decision.confidence:.2f}"
        draw.text((4, 4), text, fill=(255, 255, 0))

    bar_w = max(10, int(debug_w * 0.2))
    bar_h = 10
    bar_x, bar_y = 4, debug_h - bar_h - 4
    draw.rectangle([bar_x, bar_y, bar_x + bar_w, bar_y + bar_h], outline=(255, 255, 255))
    if in_speech:
        draw.rectangle([bar_x, bar_y, bar_x + bar_w, bar_y + bar_h], fill=(0, 200, 0))

    out = np.array(img)[:, :, ::-1]  # RGB -> BGR
    return np.ascontiguousarray(out)


def _build_main_encode_cmd(
    out_path: Path,
    crop_w: int,
    crop_h: int,
    fps: float,
    input_path: Path,
    start_s: float,
    end_s: float,
    is_segment: bool,
    crf: int,
    preset: str,
    audio_mode: str,
) -> list[str]:
    cmd = [
        "-y",
        "-v",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{crop_w}x{crop_h}",
        "-r",
        str(fps),
        "-i",
        "-",
    ]
    if is_segment:
        cmd += ["-ss", f"{start_s:.3f}", "-t", f"{end_s - start_s:.3f}"]
    cmd += ["-i", input_path.as_posix()]
    cmd += [
        "-map",
        "0:v:0",
        "-map",
        "1:a:0?",
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        audio_mode,
    ]
    if audio_mode == "aac":
        cmd += ["-b:a", "192k"]
    cmd += ["-shortest", "-movflags", "+faststart", out_path.as_posix()]
    return cmd


def _build_debug_encode_cmd(
    debug_path: Path, debug_w: int, debug_h: int, fps: float, preset: str, crf: int
) -> list[str]:
    return [
        "-y",
        "-v",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{debug_w}x{debug_h}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        debug_path.as_posix(),
    ]


def _start_encoder(cmd_args: list[str], log_file: Path) -> tuple["subprocess.Popen", TextIO]:
    require_tools()
    log_file.parent.mkdir(parents=True, exist_ok=True)
    lf = open(log_file, "a", encoding="utf-8")
    proc = subprocess.Popen(
        ["ffmpeg"] + cmd_args, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=lf
    )
    return proc, lf


def render(
    timeline_path: Union[Path, str],
    out_path: Union[Path, str],
    cfg: Config,
    debug_overlay: bool = False,
) -> Path:
    """Render RUN_DIR/output.mp4 (and optionally <stem>_debug.mp4) from a saved timeline."""
    timeline_path = Path(timeline_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    logs_dir = out_path.parent / "logs"

    timeline = load_timeline(timeline_path)
    input_path = Path(timeline.input.path)
    W, H = timeline.input.width, timeline.input.height
    crop_w, crop_h = timeline.target.crop_w, timeline.target.crop_h
    fps = timeline.target.fps
    aspect = timeline.target.aspect
    start_s, end_s = timeline.segment.start_s, timeline.segment.end_s
    is_segment = start_s > 1e-9 or (timeline.input.duration_s - end_s) > 1e-3

    crops_sorted = sorted(timeline.crops, key=lambda c: c.f)
    n_crops = len(crops_sorted)
    if n_crops == 0:
        raise ProcessingError("Timeline has no crops to render", code="empty_timeline")

    def _crop_for(i: int):
        return crops_sorted[i] if i < n_crops else crops_sorted[-1]

    # box_at() expects plain dicts (it's shared with the live tracker in analyze.py);
    # timeline.face_tracks are pydantic FaceTrack models, so convert once up front.
    tracks_by_id = {tr.track_id: tr.model_dump() for tr in timeline.face_tracks}
    sample_fps = timeline.analysis.sample_fps
    n_ticks = len(timeline.decisions)

    def _tick_idx(t: float) -> Optional[int]:
        if n_ticks == 0:
            return None
        idx = math.floor((t - start_s) * sample_fps)
        return max(0, min(idx, n_ticks - 1))

    debug_w, debug_h = _debug_dims(W, H, cfg.render.debug_width)
    debug_path = out_path.with_name(out_path.stem + "_debug.mp4")

    def _encode_pass(audio_mode: str) -> tuple[int, Optional[int]]:
        main_cmd = _build_main_encode_cmd(
            out_path,
            crop_w,
            crop_h,
            fps,
            input_path,
            start_s,
            end_s,
            is_segment,
            cfg.render.crf,
            cfg.render.preset,
            audio_mode,
        )
        main_proc, main_lf = _start_encoder(main_cmd, logs_dir / "ffmpeg_render_encode.log")
        debug_proc: Optional[subprocess.Popen] = None
        debug_lf: Optional[TextIO] = None
        if debug_overlay:
            debug_cmd = _build_debug_encode_cmd(
                debug_path, debug_w, debug_h, fps, cfg.render.preset, cfg.render.crf
            )
            debug_proc, debug_lf = _start_encoder(debug_cmd, logs_dir / "ffmpeg_debug_encode.log")

        n_violations = 0
        try:
            with FrameReader(
                input_path,
                out_w=W,
                out_h=H,
                fps=fps,
                start_s=start_s if is_segment else None,
                end_s=end_s if is_segment else None,
                log_file=logs_dir / "ffmpeg_render_decode.log",
                mode="render",
            ) as reader:
                n_reader_frames = 0
                for i, t, frame in reader:
                    n_reader_frames += 1
                    crop = _crop_for(i)

                    rect = (crop.x, crop.y, crop.w, crop.h) if crop.mode == "speaker" else (0, 0, W, H)
                    codes = check_crop_invariants(rect, W, H, aspect, crop.mode)
                    if codes:
                        n_violations += 1
                        logger.warning(
                            f"render: invariant violation at frame {i}: {codes}",
                            extra={
                                "stage": "render",
                                "event": "invariant_violation",
                                "frame": i,
                                "codes": codes,
                            },
                        )

                    out_frame = _render_frame(frame, crop, crop_w, crop_h)
                    try:
                        main_proc.stdin.write(out_frame.tobytes())  # type: ignore[union-attr]
                        if debug_proc is not None:
                            tick_idx = _tick_idx(t)
                            decision = timeline.decisions[tick_idx] if tick_idx is not None else None
                            in_speech = any(
                                seg.start <= t <= seg.end for seg in timeline.vad_segments
                            )
                            dbg_frame = _draw_debug_frame(
                                frame, debug_w, debug_h, W, H, tracks_by_id, t, decision, crop, in_speech
                            )
                            debug_proc.stdin.write(dbg_frame.tobytes())  # type: ignore[union-attr]
                    except BrokenPipeError:
                        break

                if n_reader_frames > n_crops:
                    logger.info(
                        f"render: reader produced {n_reader_frames} frames vs {n_crops} planned "
                        f"crops; reused the last crop for the extra {n_reader_frames - n_crops} frame(s)",
                        extra={
                            "stage": "render",
                            "event": "frame_count_mismatch",
                            "reader_frames": n_reader_frames,
                            "n_crops": n_crops,
                        },
                    )
        finally:
            assert main_proc.stdin is not None
            main_proc.stdin.close()
            main_rc = main_proc.wait()
            main_lf.close()

            debug_rc: Optional[int] = None
            if debug_proc is not None:
                assert debug_proc.stdin is not None
                debug_proc.stdin.close()
                debug_rc = debug_proc.wait()
                debug_lf.close()  # type: ignore[union-attr]

        if n_violations:
            logger.warning(
                f"render: {n_violations} frame(s) failed the crop-invariant re-check",
                extra={"stage": "render", "event": "invariant_violations_total", "count": n_violations},
            )

        return main_rc, debug_rc

    main_rc, debug_rc = _encode_pass("copy")
    if main_rc != 0:
        if timeline.input.has_audio:
            logger.warning(
                "render: encode with -c:a copy failed; retrying with AAC",
                extra={"stage": "render", "event": "audio_reencoded", "returncode": main_rc},
            )
            main_rc, debug_rc = _encode_pass("aac")
            if main_rc != 0:
                raise ProcessingError(
                    f"ffmpeg render encode failed (exit {main_rc}); "
                    f"see {logs_dir / 'ffmpeg_render_encode.log'}",
                    code="render_encode_failed",
                )
        else:
            raise ProcessingError(
                f"ffmpeg render encode failed (exit {main_rc}); "
                f"see {logs_dir / 'ffmpeg_render_encode.log'}",
                code="render_encode_failed",
            )

    if debug_overlay and debug_rc:
        raise ProcessingError(
            f"ffmpeg debug overlay encode failed (exit {debug_rc}); "
            f"see {logs_dir / 'ffmpeg_debug_encode.log'}",
            code="debug_encode_failed",
        )

    logger.info(f"render complete -> {out_path}", extra={"stage": "render", "event": "render_complete"})
    return out_path


__all__ = ["render"]
