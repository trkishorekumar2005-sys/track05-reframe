"""Evaluation orchestration and report writing per CONTRACT §14.

`run_eval` implements `reframe eval --runs RUNS_DIR --labels labels --out OUT_DIR`:
walk RUNS_DIR recursively for decision_timeline.json, match each run to a label
by clip filename, compute metrics, and write per_run.csv / summary.md /
summary.json to OUT_DIR. A run with no matching (or no valid) label still gets
a row with unlabelled metrics only -- this never raises.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Optional, Union

from reframe.errors import InvalidInputError
from reframe.eval.labels import Label, load_labels
from reframe.eval.metrics import (
    crop_bound_violations,
    eligible_speaker_runs,
    fallback_prf,
    jitter_and_speed,
    switch_latency,
    switches_summary,
    visible_speaker_accuracy,
)
from reframe.timeline import Timeline, load_timeline


def _mean(values: list[Optional[float]]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None

COLUMNS = [
    "run_dir",
    "clip",
    "labelled",
    "split",
    "visible_speaker_accuracy",
    "visible_speaker_ticks",
    "fallback_precision",
    "fallback_recall",
    "fallback_f1",
    "confident_wrong_crop_rate",
    "switch_latency_median_s",
    "switch_latency_p90_s",
    "switches_missed",
    "crop_bound_violations_count",
    "crop_bound_violations_pct",
    "jitter_px_per_frame2",
    "jitter_norm",
    "p95_speed_px_per_frame",
    "n_switches",
    "switches_per_min",
    "analysis_s",
    "render_s",
    "total_s",
    "clip_duration_s",
    "realtime_factor",
    "peak_rss_mb",
    "model_size_mb",
    "config_hash",
    "git_commit",
]

_LABEL_ONLY_KEYS = (
    "visible_speaker_accuracy",
    "visible_speaker_ticks",
    "fallback_precision",
    "fallback_recall",
    "fallback_f1",
    "confident_wrong_crop_rate",
    "switch_latency_median_s",
    "switch_latency_p90_s",
    "switches_missed",
)

_RUN_METRICS_KEYS = (
    "analysis_s",
    "render_s",
    "total_s",
    "clip_duration_s",
    "realtime_factor",
    "peak_rss_mb",
    "model_size_mb",
)


def evaluate_run(
    timeline: Timeline,
    label: Optional[Label],
    metrics_json: Optional[dict],
    run_dir: str,
) -> dict:
    """Compute the full CONTRACT §14 per-run metrics row for one run."""
    W = float(timeline.input.width)
    H = float(timeline.input.height)
    aspect = timeline.target.aspect
    start_s = timeline.segment.start_s
    sample_fps = timeline.analysis.sample_fps

    tracks = {ft.track_id: {"samples": [s.model_dump() for s in ft.samples]} for ft in timeline.face_tracks}
    decisions = [d.model_dump() for d in timeline.decisions]
    crops = [c.model_dump() for c in timeline.crops]

    row: dict = {
        "run_dir": run_dir,
        "clip": Path(timeline.input.path).name,
        "labelled": label is not None,
        "split": label.split if label is not None else None,
    }

    if label is not None:
        intervals = [{"start": iv.start, "end": iv.end, "expect": iv.expect} for iv in label.intervals]
        people = {name: pr.x_range for name, pr in label.people.items()}

        acc = visible_speaker_accuracy(decisions, tracks, intervals, people, label.tolerance_s, W)
        row["visible_speaker_accuracy"] = acc["value"] if acc else None
        row["visible_speaker_ticks"] = acc["total"] if acc else None

        prf = fallback_prf(decisions, intervals, label.tolerance_s)
        row["fallback_precision"] = prf["precision"] if prf else None
        row["fallback_recall"] = prf["recall"] if prf else None
        row["fallback_f1"] = prf["f1"] if prf else None
        row["confident_wrong_crop_rate"] = prf["confident_wrong_crop_rate"] if prf else None

        dt = 1.0 / sample_fps if sample_fps else 0.0
        lat = switch_latency(decisions, intervals, people, tracks, W, dt=dt)
        row["switch_latency_median_s"] = lat["median_s"]
        row["switch_latency_p90_s"] = lat["p90_s"]
        row["switches_missed"] = lat["switches_missed"]
    else:
        for k in _LABEL_ONLY_KEYS:
            row[k] = None

    cbv = crop_bound_violations(crops, W, H, aspect)
    row["crop_bound_violations_count"] = cbv["count"]
    row["crop_bound_violations_pct"] = cbv["pct"]

    runs = eligible_speaker_runs(crops, decisions, start_s, sample_fps)
    jit = jitter_and_speed(runs, W)
    row["jitter_px_per_frame2"] = jit["jitter_px_per_frame2"]
    row["jitter_norm"] = jit["jitter_norm"]
    row["p95_speed_px_per_frame"] = jit["p95_speed_px_per_frame"]

    duration_s = timeline.segment.end_s - timeline.segment.start_s
    sw = switches_summary(len(timeline.switches), duration_s)
    row["n_switches"] = sw["n_switches"]
    row["switches_per_min"] = sw["switches_per_min"]

    if metrics_json:
        for k in _RUN_METRICS_KEYS:
            row[k] = metrics_json.get(k)
        row["config_hash"] = metrics_json.get("config_hash", timeline.config_hash)
        row["git_commit"] = metrics_json.get("git_commit", timeline.tool.git_commit)
    else:
        for k in _RUN_METRICS_KEYS:
            row[k] = None
        row["config_hash"] = timeline.config_hash
        row["git_commit"] = timeline.tool.git_commit

    return row


def _find_timelines(runs_dir: Path) -> list[Path]:
    return sorted(runs_dir.rglob("decision_timeline.json"))


def run_eval(runs_dir: Union[Path, str], labels_dir: Union[Path, str], out_dir: Union[Path, str]) -> dict:
    """Walk RUNS_DIR recursively for decision_timeline.json, match each to a
    label by clip filename, and write the report to OUT_DIR. Returns a small
    summary dict {n_runs, n_labelled, warnings} for the CLI to print."""
    runs_dir = Path(runs_dir)
    out_dir = Path(out_dir)
    if not runs_dir.is_dir():
        raise InvalidInputError(f"Runs directory not found: {runs_dir}", code="dir_not_found")

    label_set = load_labels(labels_dir)

    warnings: list[str] = [f"label {name}: {msg}" for name, msg in label_set.errors.items()]

    rows: list[dict] = []
    for timeline_path in _find_timelines(runs_dir):
        run_dir = timeline_path.parent
        timeline = load_timeline(timeline_path)
        clip = Path(timeline.input.path).name
        label = label_set.by_clip.get(clip)
        if label is None:
            warnings.append(f"run {run_dir.as_posix()}: no label for clip '{clip}'")

        metrics_path = run_dir / "metrics.json"
        metrics_json = None
        if metrics_path.is_file():
            metrics_json = json.loads(metrics_path.read_text(encoding="utf-8"))

        rows.append(evaluate_run(timeline, label, metrics_json, run_dir.as_posix()))

    write_report(rows, out_dir)

    return {
        "n_runs": len(rows),
        "n_labelled": sum(1 for r in rows if r["labelled"]),
        "warnings": warnings,
    }


def write_report(rows: list[dict], out_dir: Union[Path, str]) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(rows, out_dir / "per_run.csv")
    _write_summary_json(rows, out_dir / "summary.json")
    _write_summary_md(rows, out_dir / "summary.md")


def _write_csv(rows: list[dict], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in COLUMNS})


def _write_summary_json(rows: list[dict], path: Path) -> None:
    labelled_rows = [r for r in rows if r["labelled"]]
    summary = {
        "n_runs": len(rows),
        "n_labelled": len(labelled_rows),
        "mean_visible_speaker_accuracy": _mean([r["visible_speaker_accuracy"] for r in labelled_rows]),
        "mean_fallback_f1": _mean([r["fallback_f1"] for r in labelled_rows]),
        "mean_crop_bound_violations_pct": _mean([r["crop_bound_violations_pct"] for r in rows]),
        "mean_jitter_norm": _mean([r["jitter_norm"] for r in rows]),
        "runs": rows,
    }
    path.write_text(json.dumps(summary, indent=1), encoding="utf-8")


def _fmt(v: object) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


_MD_COLUMNS = [
    "clip",
    "labelled",
    "visible_speaker_accuracy",
    "fallback_f1",
    "switch_latency_median_s",
    "crop_bound_violations_pct",
    "jitter_norm",
    "n_switches",
]


def _write_summary_md(rows: list[dict], path: Path) -> None:
    lines = [
        "# reframe eval summary",
        "",
        f"Runs: {len(rows)} (labelled: {sum(1 for r in rows if r['labelled'])})",
        "",
        "| " + " | ".join(_MD_COLUMNS) + " |",
        "|" + "|".join(["---"] * len(_MD_COLUMNS)) + "|",
    ]
    for r in rows:
        lines.append("| " + " | ".join(_fmt(r.get(h)) for h in _MD_COLUMNS) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


__all__ = ["evaluate_run", "run_eval", "write_report", "COLUMNS"]
