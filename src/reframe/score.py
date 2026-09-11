"""Per-tick speaker-confidence scoring strategies per CONTRACT §7."""

from __future__ import annotations

from typing import Callable, Optional

from reframe.audio import AudioFeatures
from reframe.config import Config
from reframe.errors import ProcessingError
from reframe.lips import window_features
from reframe.track import box_at


def _visible_boxes(tracks: list[dict], t: float) -> dict[int, list[float]]:
    boxes: dict[int, list[float]] = {}
    for tr in tracks:
        box = box_at(tr, t)
        if box is not None:
            boxes[tr["track_id"]] = box
    return boxes


def _mouth_series_by_track(
    tracks: list[dict], tick_times: list[float]
) -> dict[int, list[Optional[float]]]:
    """Each track's mouth_open aligned to the global tick_times grid (None where absent)."""
    out: dict[int, list[Optional[float]]] = {}
    for tr in tracks:
        samples_by_t = {s["t"]: s.get("mouth_open") for s in tr["samples"]}
        out[tr["track_id"]] = [samples_by_t.get(t) for t in tick_times]
    return out


def _window_ticks(cfg: Config) -> int:
    return max(1, round(cfg.speaker.window_s * cfg.analysis.sample_fps))


def _lip_norm(lip_activity: float, cfg: Config) -> float:
    norm = cfg.speaker.lip_activity_norm
    if norm <= 0.0:
        return 1.0 if lip_activity > 0.0 else 0.0
    return min(1.0, lip_activity / norm)


def _largest_face(
    tracks: list[dict],
    tick_times: list[float],
    k: int,
    cfg: Config,
    audio: Optional[AudioFeatures],
    mouth_series: dict[int, list[Optional[float]]],
) -> dict[int, float]:
    """Naive baseline: score = box area / largest visible area. Ignores audio."""
    boxes = _visible_boxes(tracks, tick_times[k])
    if not boxes:
        return {}
    areas = {tid: box[2] * box[3] for tid, box in boxes.items()}
    max_area = max(areas.values())
    if max_area <= 0.0:
        return {tid: 0.0 for tid in areas}
    return {tid: area / max_area for tid, area in areas.items()}


def _av_heuristic(
    tracks: list[dict],
    tick_times: list[float],
    k: int,
    cfg: Config,
    audio: Optional[AudioFeatures],
    mouth_series: dict[int, list[Optional[float]]],
) -> dict[int, float]:
    """`vad[k] * lip_norm` per CONTRACT §7, per visible track."""
    if audio is None:
        raise ProcessingError(
            "Scoring strategy 'av_heuristic' requires audio features", code="missing_audio_features"
        )
    boxes = _visible_boxes(tracks, tick_times[k])
    if not boxes:
        return {}

    vad_k = audio.vad[k] if k < len(audio.vad) else 0.0
    window_ticks = _window_ticks(cfg)
    n = len(tick_times)

    scores: dict[int, float] = {}
    for tid in boxes:
        series = mouth_series.get(tid) or [None] * n
        lip_activity, _ = window_features(series, audio.energy, k, window_ticks)
        scores[tid] = vad_k * _lip_norm(lip_activity, cfg)
    return scores


def _av_corr(
    tracks: list[dict],
    tick_times: list[float],
    k: int,
    cfg: Config,
    audio: Optional[AudioFeatures],
    mouth_series: dict[int, list[Optional[float]]],
) -> dict[int, float]:
    """`vad[k] * max(0, av_corr)` per CONTRACT §7, per visible track."""
    if audio is None:
        raise ProcessingError(
            "Scoring strategy 'av_corr' requires audio features", code="missing_audio_features"
        )
    boxes = _visible_boxes(tracks, tick_times[k])
    if not boxes:
        return {}

    vad_k = audio.vad[k] if k < len(audio.vad) else 0.0
    window_ticks = _window_ticks(cfg)
    n = len(tick_times)

    scores: dict[int, float] = {}
    for tid in boxes:
        series = mouth_series.get(tid) or [None] * n
        _, av_corr = window_features(series, audio.energy, k, window_ticks)
        scores[tid] = vad_k * max(0.0, av_corr)
    return scores


STRATEGIES: dict[str, Callable[..., dict[int, float]]] = {
    "largest_face": _largest_face,
    "av_heuristic": _av_heuristic,
    "av_corr": _av_corr,
}


def score_ticks(
    tracks: list[dict],
    tick_times: list[float],
    strategy: str,
    cfg: Config,
    audio: Optional[AudioFeatures] = None,
) -> list[dict[int, float]]:
    """Compute per-tick {track_id: score} using the configured strategy."""
    if strategy not in STRATEGIES:
        raise ProcessingError(f"Unknown scoring strategy '{strategy}'", code="unknown_strategy")
    fn = STRATEGIES[strategy]
    mouth_series = _mouth_series_by_track(tracks, tick_times)
    return [fn(tracks, tick_times, k, cfg, audio, mouth_series) for k in range(len(tick_times))]


__all__ = ["STRATEGIES", "score_ticks"]
