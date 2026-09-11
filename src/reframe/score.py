"""Per-tick speaker-confidence scoring strategies per CONTRACT §7."""

from __future__ import annotations

from typing import Any, Callable, Optional

from reframe.config import Config
from reframe.errors import ProcessingError
from reframe.track import box_at


def _visible_boxes(tracks: list[dict], t: float) -> dict[int, list[float]]:
    boxes: dict[int, list[float]] = {}
    for tr in tracks:
        box = box_at(tr, t)
        if box is not None:
            boxes[tr["track_id"]] = box
    return boxes


def _largest_face(
    tracks: list[dict], t: float, cfg: Config, audio: Optional[Any] = None
) -> dict[int, float]:
    """Naive baseline: score = box area / largest visible area. Ignores audio."""
    boxes = _visible_boxes(tracks, t)
    if not boxes:
        return {}
    areas = {tid: box[2] * box[3] for tid, box in boxes.items()}
    max_area = max(areas.values())
    if max_area <= 0.0:
        return {tid: 0.0 for tid in areas}
    return {tid: area / max_area for tid, area in areas.items()}


def _av_heuristic(
    tracks: list[dict], t: float, cfg: Config, audio: Optional[Any] = None
) -> dict[int, float]:
    raise ProcessingError(
        "Scoring strategy 'av_heuristic' is not implemented yet",
        code="strategy_not_implemented",
    )


def _av_corr(
    tracks: list[dict], t: float, cfg: Config, audio: Optional[Any] = None
) -> dict[int, float]:
    raise ProcessingError(
        "Scoring strategy 'av_corr' is not implemented yet",
        code="strategy_not_implemented",
    )


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
    audio: Optional[Any] = None,
) -> list[dict[int, float]]:
    """Compute per-tick {track_id: score} using the configured strategy."""
    if strategy not in STRATEGIES:
        raise ProcessingError(f"Unknown scoring strategy '{strategy}'", code="unknown_strategy")
    fn = STRATEGIES[strategy]
    return [fn(tracks, t, cfg, audio) for t in tick_times]


__all__ = ["STRATEGIES", "score_ticks"]
