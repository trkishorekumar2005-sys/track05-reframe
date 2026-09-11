"""Analysis pipeline per CONTRACT §2, §13: probe -> detect -> track -> score -> decide -> crop."""

from __future__ import annotations

import dataclasses
import datetime
import logging
import time
from pathlib import Path
from typing import Optional, Union

from reframe import __version__
from reframe.audio import AudioFeatures, compute_audio_features
from reframe.cache import StageCache
from reframe.config import Config, config_hash
from reframe.crop import crop_size, plan_crops
from reframe.decide import build_fallback_periods, build_speaker_segments, decide
from reframe.detect import make_detector
from reframe.ffmpeg_utils import FrameReader
from reframe.lips import make_mouth_analyzer
from reframe.probe import probe
from reframe.scene import SceneAnalyzer
from reframe.score import score_ticks
from reframe.timeline import (
    AnalysisInfo,
    FaceTrack,
    ModelEntry,
    SegmentInfo,
    TargetInfo,
    Timeline,
    TimingInfo,
    ToolInfo,
    save_timeline,
)
from reframe.track import IoUTracker

logger = logging.getLogger("reframe")

_MODELS_LOCK_PATH = Path(__file__).resolve().parent.parent.parent / "models" / "manifest.lock.json"


def _analysis_dims(width: int, height: int, analysis_width: int) -> tuple[int, int, float]:
    """Analysis frame size per CONTRACT §4: never upscale width; even height."""
    out_w = min(analysis_width, width)
    scale = out_w / width
    out_h = int(round(height * scale))
    if out_h % 2 != 0:
        out_h += 1
    return out_w, out_h, scale


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


def _attach_scores_to_tracks(
    tracks: list[dict], tick_times: list[float], scores_per_tick: list[dict[int, float]]
) -> None:
    """Write each tick's {track_id: score} onto that track's sample (in place)."""
    samples_by_track: dict[int, dict[float, dict]] = {
        tr["track_id"]: {s["t"]: s for s in tr["samples"]} for tr in tracks
    }
    for k, t in enumerate(tick_times):
        for tid, score in scores_per_tick[k].items():
            sample = samples_by_track.get(tid, {}).get(t)
            if sample is not None:
                sample["score"] = round(float(score), 3)


def _models_info(names: list[str]) -> list[dict]:
    import json

    if not _MODELS_LOCK_PATH.is_file():
        return []
    with open(_MODELS_LOCK_PATH, "r", encoding="utf-8") as f:
        lock = json.load(f)
    out = []
    for name in names:
        entry = lock.get(name)
        if entry is None:
            continue
        out.append(
            {
                "name": name,
                "file": entry["filename"],
                "sha256": entry["sha256"],
                "size_mb": entry["size_mb"],
            }
        )
    return out


def analyze(
    input_path: Union[Path, str],
    aspect: str,
    out_dir: Union[Path, str],
    cfg: Config,
    start_s: Optional[float] = None,
    end_s: Optional[float] = None,
    resume: bool = False,
) -> Path:
    """Run the analysis pipeline and write RUN_DIR/decision_timeline.json. Returns its path."""
    t_analysis_start = time.perf_counter()

    input_path = Path(input_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --aspect overrides crop.aspect per CONTRACT §16.
    cfg = cfg.model_copy(deep=True)
    cfg.crop.aspect = aspect

    info = probe(input_path, cfg)
    eff_start = start_s if start_s is not None else 0.0
    eff_end = end_s if end_s is not None else info.duration_s

    out_w, out_h, scale = _analysis_dims(info.width, info.height, cfg.analysis.analysis_width)
    tick_times = _tick_times(eff_start, eff_end, cfg.analysis.sample_fps)
    n_ticks = len(tick_times)

    cfg_hash = config_hash(cfg)
    cache = StageCache(cfg.runtime.cache_dir, info.sha256, cfg_hash, start_s, end_s)

    # --- Vision stage (detect + track + lips) and scene stage, cached ---
    # Computed together in one frame-decode pass but cached as separate stage
    # blobs, matching cache.STAGES = ("vision", "audio", "scene").
    t_vision_start = time.perf_counter()
    cached_vision = cache.load("vision") if resume else None
    cached_scene = cache.load("scene") if resume else None
    if cached_vision is not None and cached_scene is not None:
        tracks = cached_vision["tracks"]
        is_cut_ticks = cached_scene["is_cut"]
        is_screen_content_ticks = cached_scene["is_screen_content"]
    else:
        detector = make_detector(cfg, scale)
        tracker = IoUTracker(cfg)
        mouth_analyzer = make_mouth_analyzer(cfg, scale)
        scene_analyzer = SceneAnalyzer(cfg)
        # Fixed-size, tick-indexed (not append-as-decoded): FrameReader's real
        # decoded frame count can be off by one from the tick_times formula
        # near the clip's tail, so index by `i` and leave any missing tail
        # tick at its safe default rather than shortening these arrays.
        is_cut_ticks: list[bool] = [False] * n_ticks
        is_screen_content_ticks: list[bool] = [False] * n_ticks
        try:
            with FrameReader(
                input_path,
                out_w=out_w,
                out_h=out_h,
                fps=cfg.analysis.sample_fps,
                start_s=eff_start,
                end_s=eff_end,
                log_file=out_dir / "logs" / "ffmpeg_analysis.log",
                mode="analysis",
            ) as reader:
                for i, t, frame in reader:
                    t_ms = int(round(t * 1000))
                    detections = detector.detect(frame, t_ms)
                    tracker.update(t, detections)

                    samples_this_tick = tracker.current_samples(t)
                    track_boxes = {tid: s["bbox"] for tid, s in samples_this_tick.items()}
                    mouth_by_track = mouth_analyzer.analyze(frame, t_ms, track_boxes)
                    for tid, mouth_open in mouth_by_track.items():
                        if tid in samples_this_tick:
                            samples_this_tick[tid]["mouth_open"] = round(float(mouth_open), 3)

                    if i < n_ticks:
                        is_cut, is_screen_content = scene_analyzer.analyze(
                            frame, n_faces=len(detections)
                        )
                        is_cut_ticks[i] = is_cut
                        is_screen_content_ticks[i] = is_screen_content
        finally:
            detector.close()
            mouth_analyzer.close()
        tracks = tracker.finalize()
        cache.save("vision", {"tracks": tracks})
        cache.save("scene", {"is_cut": is_cut_ticks, "is_screen_content": is_screen_content_ticks})
    vision_s = time.perf_counter() - t_vision_start
    logger.info(
        f"vision stage done in {vision_s:.3f}s ({len(tracks)} confirmed tracks)",
        extra={"stage": "vision", "event": "stage_complete", "duration_s": vision_s},
    )

    # --- Audio stage (VAD + energy), cached ---
    t_audio_start = time.perf_counter()
    cached_audio = cache.load("audio") if resume else None
    if cached_audio is not None:
        audio_features = AudioFeatures(**cached_audio)
    else:
        audio_features = compute_audio_features(
            info, cfg, tick_times, eff_start, eff_end, out_dir / "logs"
        )
        cache.save("audio", dataclasses.asdict(audio_features))
    audio_s = time.perf_counter() - t_audio_start
    logger.info(
        f"audio stage done in {audio_s:.3f}s",
        extra={"stage": "audio", "event": "stage_complete", "duration_s": audio_s},
    )

    # --- Scoring ---
    t_score_start = time.perf_counter()
    scores_per_tick = score_ticks(tracks, tick_times, cfg.speaker.strategy, cfg, audio=audio_features)
    _attach_scores_to_tracks(tracks, tick_times, scores_per_tick)
    score_s = time.perf_counter() - t_score_start
    logger.info(
        f"scoring done in {score_s:.3f}s",
        extra={"stage": "score", "event": "stage_complete", "duration_s": score_s},
    )

    vad_ticks = audio_features.vad

    # --- Decide ---
    t_decide_start = time.perf_counter()
    ticks = [
        {
            "t": tick_times[i],
            "visible_ids": list(scores_per_tick[i].keys()),
            "scores": scores_per_tick[i],
            "vad": vad_ticks[i],
            "is_cut": is_cut_ticks[i],
            "is_screen_content": is_screen_content_ticks[i],
        }
        for i in range(n_ticks)
    ]
    decisions, switches = decide(ticks, cfg, has_audio=info.has_audio)
    speaker_segments = build_speaker_segments(decisions)
    fallback_periods = build_fallback_periods(decisions)
    decide_s = time.perf_counter() - t_decide_start
    logger.info(
        f"decide done in {decide_s:.3f}s ({len(switches)} switches)",
        extra={"stage": "decide", "event": "stage_complete", "duration_s": decide_s},
    )

    scene_cuts = [t for t, cut in zip(tick_times, is_cut_ticks) if cut]
    vad_segs = audio_features.vad_segments

    # --- Crop planning ---
    t_crop_start = time.perf_counter()
    crop_w, crop_h = crop_size(info.width, info.height, cfg.crop.aspect)
    n_frames = round((eff_end - eff_start) * info.fps)
    crops, violations = plan_crops(
        decisions=decisions,
        tracks=tracks,
        W=info.width,
        H=info.height,
        fps=info.fps,
        start_s=eff_start,
        n_frames=n_frames,
        sample_fps=cfg.analysis.sample_fps,
        scene_cuts=scene_cuts,
        cfg=cfg,
    )
    crop_s = time.perf_counter() - t_crop_start

    by_code: dict[str, int] = {}
    for v in violations:
        for code in v["codes"]:
            by_code[code] = by_code.get(code, 0) + 1
    violation_count = sum(by_code.values())
    logger.info(
        f"crop planning done in {crop_s:.3f}s ({len(crops)} frames, {violation_count} invariant violations)",
        extra={
            "stage": "crop",
            "event": "stage_complete",
            "duration_s": crop_s,
            "violation_count": violation_count,
        },
    )

    warnings = list(dict.fromkeys(info.warnings + audio_features.warnings))

    analysis_s = time.perf_counter() - t_analysis_start

    timeline = Timeline(
        tool=ToolInfo(name="reframe", version=__version__, git_commit=None),
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        input=info,
        target=TargetInfo(aspect=cfg.crop.aspect, crop_w=crop_w, crop_h=crop_h, fps=info.fps),
        segment=SegmentInfo(start_s=eff_start, end_s=eff_end),
        config_hash=cfg_hash,
        config=cfg,
        models=[ModelEntry(**m) for m in _models_info(["blaze_face_short_range", "face_landmarker"])],
        analysis=AnalysisInfo(
            sample_fps=cfg.analysis.sample_fps,
            analysis_width=out_w,
            analysis_height=out_h,
            detector=cfg.analysis.detector,
            strategy=cfg.speaker.strategy,
            n_ticks=n_ticks,
        ),
        face_tracks=[FaceTrack(**tr) for tr in tracks],
        vad_segments=vad_segs,
        scene_cuts=scene_cuts,
        decisions=decisions,
        speaker_segments=speaker_segments,
        switches=switches,
        fallback_periods=fallback_periods,
        crops=crops,
        invariant_violations={"count": violation_count, "by_code": by_code, "examples": violations[:20]},
        warnings=warnings,
        timing=TimingInfo(analysis_s=round(analysis_s, 3)),
    )

    timeline_path = out_dir / "decision_timeline.json"
    save_timeline(timeline, timeline_path)
    logger.info(
        f"analysis complete in {analysis_s:.3f}s -> {timeline_path}",
        extra={"stage": "analyze", "event": "analysis_complete", "duration_s": analysis_s},
    )
    return timeline_path


__all__ = ["analyze"]
