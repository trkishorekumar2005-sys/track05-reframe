"""Stage caching per CONTRACT §13."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional, Union

logger = logging.getLogger("reframe")

STAGES = ("vision", "audio", "scene")


def _fmt_bound(x: Optional[float]) -> str:
    return "full" if x is None else f"{x:.3f}"


class StageCache:
    """Per-input, per-config, per-segment JSON cache for pipeline stages.

    Layout: {cache_dir}/{input_sha[:16]}/{config_hash}_{start}_{end}/{stage}.json
    """

    def __init__(
        self,
        cache_dir: Union[Path, str],
        input_sha: str,
        config_hash: str,
        start_s: Optional[float],
        end_s: Optional[float],
    ):
        self._dir = (
            Path(cache_dir)
            / input_sha[:16]
            / f"{config_hash}_{_fmt_bound(start_s)}_{_fmt_bound(end_s)}"
        )

    def path_for(self, stage: str) -> Path:
        return self._dir / f"{stage}.json"

    def load(self, stage: str) -> Optional[Any]:
        """Return the cached value for `stage`, or None if not cached."""
        p = self.path_for(stage)
        if not p.is_file():
            return None
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info(f"cache hit for stage '{stage}'", extra={"stage": stage, "event": "cache_hit"})
        return data

    def save(self, stage: str, data: Any) -> None:
        """Write `data` to the cache for `stage`."""
        p = self.path_for(stage)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f)
        logger.info(f"cache miss for stage '{stage}'; wrote cache", extra={"stage": stage, "event": "cache_miss"})


__all__ = ["StageCache", "STAGES"]
