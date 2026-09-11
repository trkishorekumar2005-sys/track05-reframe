"""Audio feature extraction per CONTRACT §2, §7: VAD + energy from the original audio."""

from __future__ import annotations

import dataclasses
import subprocess
from pathlib import Path
from typing import Optional, Union

import numpy as np
import webrtcvad

from reframe.config import Config
from reframe.errors import ProcessingError
from reframe.ffmpeg_utils import require_tools
from reframe.probe import VideoInfo

_SAMPLE_RATE = 16000
_VAD_FRAME_MS = 30
_VAD_SMOOTH_S = 0.3


@dataclasses.dataclass
class AudioFeatures:
    """Per-tick audio signals for the analysis window, plus derived speech segments."""

    vad: list[float]
    energy: list[float]
    vad_segments: list[dict]
    warnings: list[str] = dataclasses.field(default_factory=list)


def _extract_pcm(input_path: Path, start_s: float, end_s: float, log_file: Path) -> np.ndarray:
    """Mono 16kHz s16le PCM samples via ffmpeg, for [start_s, end_s)."""
    require_tools()
    dur = max(0.0, end_s - start_s)
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-ss",
        f"{start_s:.3f}",
        "-t",
        f"{dur:.3f}",
        "-i",
        input_path.as_posix(),
        "-ac",
        "1",
        "-ar",
        str(_SAMPLE_RATE),
        "-f",
        "s16le",
        "-",
    ]
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with open(log_file, "a", encoding="utf-8") as lf:
        res = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=lf, timeout=max(30.0, dur * 3.0)
        )
    if res.returncode != 0:
        raise ProcessingError(
            f"ffmpeg audio extraction failed (exit {res.returncode}); see {log_file}",
            code="audio_extraction_failed",
        )
    return np.frombuffer(res.stdout, dtype="<i2")


def _frame_speech_flags(pcm_i16: np.ndarray, aggressiveness: int) -> np.ndarray:
    """webrtcvad speech/non-speech flag per 30ms frame."""
    vad = webrtcvad.Vad(aggressiveness)
    frame_len = int(_SAMPLE_RATE * _VAD_FRAME_MS / 1000)
    n_frames = len(pcm_i16) // frame_len
    if n_frames == 0:
        return np.zeros(0, dtype=bool)

    pcm_bytes = pcm_i16[: n_frames * frame_len].tobytes()
    frame_bytes_len = frame_len * 2
    flags = np.zeros(n_frames, dtype=bool)
    for i in range(n_frames):
        chunk = pcm_bytes[i * frame_bytes_len : (i + 1) * frame_bytes_len]
        flags[i] = vad.is_speech(chunk, _SAMPLE_RATE)
    return flags


def _tick_window(t_k: float, sample_fps: float) -> tuple[float, float]:
    half = 0.5 / sample_fps
    return t_k - half, t_k + half


def _vad_fraction_per_tick(
    flags: np.ndarray, start_s: float, tick_times: list[float], sample_fps: float
) -> np.ndarray:
    n_ticks = len(tick_times)
    if len(flags) == 0:
        return np.zeros(n_ticks, dtype=float)

    frame_dt = _VAD_FRAME_MS / 1000.0
    frame_centers = start_s + (np.arange(len(flags)) + 0.5) * frame_dt

    out = np.zeros(n_ticks, dtype=float)
    for k, t_k in enumerate(tick_times):
        lo, hi = _tick_window(t_k, sample_fps)
        mask = (frame_centers >= lo) & (frame_centers < hi)
        if mask.any():
            out[k] = float(flags[mask].mean())
    return out


def _energy_per_tick(
    pcm_i16: np.ndarray, start_s: float, tick_times: list[float], sample_fps: float
) -> np.ndarray:
    n_ticks = len(tick_times)
    samples = pcm_i16.astype(np.float64) / 32768.0
    n = len(samples)

    raw = np.zeros(n_ticks, dtype=float)
    for k, t_k in enumerate(tick_times):
        lo, hi = _tick_window(t_k, sample_fps)
        i0 = max(0, int(round((lo - start_s) * _SAMPLE_RATE)))
        i1 = min(n, int(round((hi - start_s) * _SAMPLE_RATE)))
        if i1 > i0:
            window = samples[i0:i1]
            raw[k] = float(np.sqrt(np.mean(window * window)))

    if n_ticks == 0:
        return raw

    p95 = float(np.percentile(raw, 95))
    if p95 > 1e-9:
        return np.clip(raw / p95, 0.0, 1.0)
    return np.zeros(n_ticks, dtype=float)


def _moving_average(x: np.ndarray, window_s: float, sample_fps: float) -> np.ndarray:
    """Centred moving average over a `window_s`-wide (in ticks) span; shrinks at the edges."""
    n = max(1, int(round(window_s * sample_fps)))
    if n <= 1 or len(x) == 0:
        return x.astype(float, copy=True)

    half = n // 2
    out = np.empty_like(x, dtype=float)
    for i in range(len(x)):
        lo = max(0, i - half)
        hi = min(len(x), i + (n - half))
        out[i] = float(x[lo:hi].mean())
    return out


def _build_vad_segments(tick_times: list[float], vad: list[float], threshold: float) -> list[dict]:
    segments: list[dict] = []
    start_idx: Optional[int] = None
    for i, v in enumerate(vad):
        speech = v >= threshold
        if speech:
            if start_idx is None:
                start_idx = i
        elif start_idx is not None:
            segments.append({"start": tick_times[start_idx], "end": tick_times[i - 1]})
            start_idx = None
    if start_idx is not None:
        segments.append({"start": tick_times[start_idx], "end": tick_times[-1]})
    return segments


def compute_audio_features(
    info: VideoInfo,
    cfg: Config,
    tick_times: list[float],
    start_s: float,
    end_s: float,
    log_dir: Union[Path, str],
) -> AudioFeatures:
    """Per-tick VAD + energy for [start_s, end_s) per CONTRACT §7. Zeros + warning if no audio."""
    n_ticks = len(tick_times)

    if not info.has_audio:
        return AudioFeatures(
            vad=[0.0] * n_ticks,
            energy=[0.0] * n_ticks,
            vad_segments=[],
            warnings=["no_audio"],
        )

    if n_ticks == 0:
        return AudioFeatures(vad=[], energy=[], vad_segments=[], warnings=[])

    sample_fps = float(cfg.analysis.sample_fps)
    log_file = Path(log_dir) / "ffmpeg_audio.log"
    pcm_i16 = _extract_pcm(Path(info.path), start_s, end_s, log_file)

    if len(pcm_i16) == 0:
        vad_raw = np.zeros(n_ticks, dtype=float)
        energy = np.zeros(n_ticks, dtype=float)
    else:
        flags = _frame_speech_flags(pcm_i16, cfg.audio.vad_aggressiveness)
        vad_raw = _vad_fraction_per_tick(flags, start_s, tick_times, sample_fps)
        energy = _energy_per_tick(pcm_i16, start_s, tick_times, sample_fps)

    vad = np.clip(_moving_average(vad_raw, _VAD_SMOOTH_S, sample_fps), 0.0, 1.0)
    vad_segments = _build_vad_segments(tick_times, vad.tolist(), cfg.audio.speech_threshold)

    return AudioFeatures(
        vad=[round(float(v), 3) for v in vad],
        energy=[round(float(e), 3) for e in energy],
        vad_segments=vad_segments,
        warnings=[],
    )


__all__ = ["AudioFeatures", "compute_audio_features"]
