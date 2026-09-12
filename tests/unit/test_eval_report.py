"""Integration-style tests for `reframe.eval.report` per CONTRACT §14:
walking RUNS_DIR recursively, matching labels by clip filename, and writing
per_run.csv / summary.md / summary.json (runs without a label still get
unlabelled metrics)."""

import csv
import json
from pathlib import Path

from reframe.eval.report import run_eval


def _timeline_dict(clip_name: str) -> dict:
    return {
        "schema_version": "1.0",
        "tool": {"name": "reframe", "version": "0.1.0", "git_commit": "abc123"},
        "created_utc": "2026-01-01T00:00:00+00:00",
        "input": {
            "path": f"data/{clip_name}",
            "sha256": "a" * 64,
            "size_bytes": 100,
            "width": 100,
            "height": 100,
            "fps": 10.0,
            "is_vfr": False,
            "duration_s": 1.0,
            "rotation": 0,
            "video_codec": "h264",
            "has_audio": True,
            "audio_codec": "aac",
            "n_frames_expected": 10,
            "warnings": [],
        },
        "target": {"aspect": "1:1", "crop_w": 100, "crop_h": 100, "fps": 10.0},
        "segment": {"start_s": 0.0, "end_s": 1.0},
        "config_hash": "0123456789abcdef",
        "config": {},
        "models": [],
        "analysis": {
            "sample_fps": 10.0,
            "analysis_width": 100,
            "analysis_height": 100,
            "detector": "mediapipe",
            "strategy": "largest_face",
            "n_ticks": 1,
        },
        "face_tracks": [
            {
                "track_id": 1,
                "first_t": 0.0,
                "last_t": 0.0,
                "samples": [{"t": 0.0, "bbox": [0.0, 0.0, 10.0, 10.0], "det_conf": 0.9, "mouth_open": None, "score": None}],
            }
        ],
        "vad_segments": [],
        "scene_cuts": [],
        "decisions": [{"t": 0.0, "mode": "speaker", "track_id": 1, "confidence": 0.9, "reason": "largest_face", "scores": {1: 0.9}}],
        "speaker_segments": [{"start": 0.0, "end": 0.0, "mode": "speaker", "track_id": 1, "mean_confidence": 0.9, "reason": "largest_face"}],
        "switches": [{"t": 0.0, "from": "FALLBACK", "to": 1, "reason": "largest_face", "confidence": 0.9}],
        "fallback_periods": [],
        "crops": [{"f": 0, "t": 0.0, "x": 0, "y": 0, "w": 100, "h": 100, "mode": "speaker"}],
        "invariant_violations": {"count": 0, "by_code": {}, "examples": []},
        "warnings": [],
        "timing": {"analysis_s": 1.5},
    }


def _label_dict(clip_name: str) -> dict:
    return {
        "clip": clip_name,
        "split": "test",
        "people": {"A": {"x_range": [0.0, 1.0]}},
        # start is well before the run's only tick (t=0.0) so it stays inside
        # the window even after the tolerance_s cutoff is applied.
        "intervals": [{"start": -0.5, "end": 1.0, "expect": "A"}],
        "tolerance_s": 0.1,
    }


def test_run_eval_walks_recursively_and_handles_unlabelled_runs(tmp_path: Path):
    runs_dir = tmp_path / "runs"
    labels_dir = tmp_path / "labels"
    out_dir = tmp_path / "out"

    run_a = runs_dir / "run_a"
    run_a.mkdir(parents=True)
    (run_a / "decision_timeline.json").write_text(json.dumps(_timeline_dict("clip_a.mp4")), encoding="utf-8")
    (run_a / "metrics.json").write_text(
        json.dumps({"analysis_s": 2.0, "render_s": 1.0, "total_s": 3.0, "clip_duration_s": 1.0,
                     "realtime_factor": 3.0, "peak_rss_mb": 512.0, "model_size_mb": 1.0,
                     "config_hash": "0123456789abcdef", "git_commit": "abc123"}),
        encoding="utf-8",
    )

    run_b = runs_dir / "nested" / "run_b"
    run_b.mkdir(parents=True)
    (run_b / "decision_timeline.json").write_text(json.dumps(_timeline_dict("clip_b.mp4")), encoding="utf-8")
    # no metrics.json and no label for run_b

    labels_dir.mkdir(parents=True)
    (labels_dir / "clip_a.json").write_text(json.dumps(_label_dict("clip_a.mp4")), encoding="utf-8")

    result = run_eval(runs_dir, labels_dir, out_dir)

    assert result["n_runs"] == 2
    assert result["n_labelled"] == 1
    assert any("clip_b.mp4" in w for w in result["warnings"])

    assert (out_dir / "per_run.csv").is_file()
    assert (out_dir / "summary.md").is_file()
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["n_runs"] == 2
    assert summary["n_labelled"] == 1

    with open(out_dir / "per_run.csv", newline="", encoding="utf-8") as f:
        rows = {row["clip"]: row for row in csv.DictReader(f)}

    assert rows["clip_a.mp4"]["labelled"] == "True"
    assert rows["clip_a.mp4"]["visible_speaker_accuracy"] == "1.0"
    assert rows["clip_a.mp4"]["analysis_s"] == "2.0"

    assert rows["clip_b.mp4"]["labelled"] == "False"
    assert rows["clip_b.mp4"]["visible_speaker_accuracy"] == ""
    assert rows["clip_b.mp4"]["analysis_s"] == ""


def test_run_eval_no_runs_still_writes_empty_report(tmp_path: Path):
    runs_dir = tmp_path / "empty_runs"
    runs_dir.mkdir()
    out_dir = tmp_path / "out"

    result = run_eval(runs_dir, tmp_path / "labels", out_dir)

    assert result["n_runs"] == 0
    assert (out_dir / "per_run.csv").is_file()
    assert (out_dir / "summary.json").is_file()
