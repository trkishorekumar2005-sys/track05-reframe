"""Plot per-tick vad/energy and per-track mouth_open/score for a run directory.

Reads RUN_DIR/decision_timeline.json plus the cached "audio" and "vision" stage
blobs that `analyze()` wrote under runtime.cache_dir for this run, and writes
RUN_DIR/signals.png. Run via `uv run python scripts/plot_signals.py RUN_DIR`
from the same working directory the run itself used (cache paths are relative).
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reframe.timeline import load_timeline


def _fmt_bound(x: Optional[float]) -> str:
    return "full" if x is None else f"{x:.3f}"


def _find_cache_dir(
    cache_root: Path, input_sha: str, config_hash: str, seg_start: float, seg_end: float
) -> Optional[Path]:
    """Best-effort match of analyze()'s StageCache layout from a finished timeline.

    decision_timeline.json only stores the *effective* segment bounds, never
    whether --start/--end were actually passed, so both the "whole clip" and
    "explicit bounds" cache-directory spellings are tried.
    """
    base = cache_root / input_sha[:16]
    candidates = [
        base / f"{config_hash}_full_full",
        base / f"{config_hash}_{_fmt_bound(seg_start)}_{_fmt_bound(seg_end)}",
    ]
    for c in candidates:
        if c.is_dir():
            return c
    return None


def _tick_times(start_s: float, end_s: float, sample_fps: float) -> list[float]:
    times = []
    k = 0
    while True:
        t = start_s + k / sample_fps
        if t >= end_s:
            break
        times.append(round(t, 3))
        k += 1
    return times


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot vad/energy/mouth_open/score signals for a reframe run."
    )
    parser.add_argument("run_dir", type=Path, help="Path to a reframe run directory.")
    args = parser.parse_args()

    run_dir: Path = args.run_dir
    timeline_path = run_dir / "decision_timeline.json"
    if not timeline_path.is_file():
        print(f"ERROR: {timeline_path} not found", file=sys.stderr)
        sys.exit(1)

    timeline = load_timeline(timeline_path)
    tick_times = _tick_times(
        timeline.segment.start_s, timeline.segment.end_s, timeline.analysis.sample_fps
    )

    cache_root = Path(timeline.config.runtime.cache_dir)
    cache_dir = _find_cache_dir(
        cache_root,
        timeline.input.sha256,
        timeline.config_hash,
        timeline.segment.start_s,
        timeline.segment.end_s,
    )
    if cache_dir is None:
        print(
            f"ERROR: could not find a stage cache for this run under {cache_root} "
            "(it may have been cleared, or this script is running from a "
            "different working directory than the run used)",
            file=sys.stderr,
        )
        sys.exit(1)

    with open(cache_dir / "audio.json", "r", encoding="utf-8") as f:
        audio = json.load(f)
    with open(cache_dir / "vision.json", "r", encoding="utf-8") as f:
        vision = json.load(f)

    vad = audio["vad"]
    energy = audio["energy"]

    # Per-track mouth_open series, aligned to tick_times by exact sample time.
    mouth_series: dict[int, list[Optional[float]]] = {}
    for tr in vision["tracks"]:
        tid = tr["track_id"]
        samples_by_t = {s["t"]: s.get("mouth_open") for s in tr["samples"]}
        mouth_series[tid] = [samples_by_t.get(t) for t in tick_times]

    # Per-track score series, straight from the finalized decisions (scores
    # aren't persisted onto track samples, only into each tick's decision).
    score_series: dict[int, list[Optional[float]]] = {
        tid: [None] * len(tick_times) for tid in mouth_series
    }
    for k, decision in enumerate(timeline.decisions):
        for tid, score in decision.scores.items():
            score_series.setdefault(tid, [None] * len(tick_times))[k] = score

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

    axes[0].plot(tick_times, vad, color="tab:blue")
    axes[0].set_ylabel("vad")
    axes[0].set_ylim(-0.05, 1.05)

    axes[1].plot(tick_times, energy, color="tab:orange")
    axes[1].set_ylabel("energy")
    axes[1].set_ylim(-0.05, 1.05)

    for tid, series in sorted(mouth_series.items()):
        axes[2].plot(tick_times, series, label=f"track {tid}", marker=".", linestyle="-")
    axes[2].set_ylabel("mouth_open")
    if mouth_series:
        axes[2].legend(fontsize="small")

    for tid, series in sorted(score_series.items()):
        axes[3].plot(tick_times, series, label=f"track {tid}", marker=".", linestyle="-")
    axes[3].set_ylabel("score")
    axes[3].set_xlabel("time (s)")
    if score_series:
        axes[3].legend(fontsize="small")

    fig.suptitle(f"reframe signals -- {run_dir}")
    fig.tight_layout()

    out_path = run_dir / "signals.png"
    fig.savefig(out_path, dpi=120)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
