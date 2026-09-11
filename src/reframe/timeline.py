"""decision_timeline.json schema and I/O per CONTRACT §10."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Union

from pydantic import BaseModel, ConfigDict, Field

from reframe.config import Config
from reframe.probe import VideoInfo


class ToolInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    version: str
    git_commit: Optional[str] = None


class TargetInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")
    aspect: str
    crop_w: int
    crop_h: int
    fps: float


class SegmentInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")
    start_s: float
    end_s: float


class ModelEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    file: str
    sha256: str
    size_mb: float


class AnalysisInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sample_fps: float
    analysis_width: int
    analysis_height: int
    detector: str
    strategy: str
    n_ticks: int


class TrackSample(BaseModel):
    model_config = ConfigDict(extra="forbid")
    t: float
    bbox: list[float]
    det_conf: float
    mouth_open: Optional[float] = None
    score: Optional[float] = None


class FaceTrack(BaseModel):
    model_config = ConfigDict(extra="forbid")
    track_id: int
    first_t: float
    last_t: float
    samples: list[TrackSample]


class VadSegment(BaseModel):
    model_config = ConfigDict(extra="forbid")
    start: float
    end: float


class Decision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    t: float
    mode: str
    track_id: Optional[int] = None
    confidence: float
    reason: str
    scores: dict[int, float] = Field(default_factory=dict)


class SpeakerSegment(BaseModel):
    model_config = ConfigDict(extra="forbid")
    start: float
    end: float
    mode: str
    track_id: Optional[int] = None
    mean_confidence: float
    reason: str


class Switch(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    t: float
    from_: Union[int, str] = Field(alias="from")
    to: Union[int, str]
    reason: str
    confidence: float


class FallbackPeriod(BaseModel):
    model_config = ConfigDict(extra="forbid")
    start: float
    end: float
    reason: str


class Crop(BaseModel):
    model_config = ConfigDict(extra="forbid")
    f: int
    t: float
    x: int
    y: int
    w: int
    h: int
    mode: str


class InvariantViolations(BaseModel):
    model_config = ConfigDict(extra="forbid")
    count: int
    by_code: dict[str, int] = Field(default_factory=dict)
    examples: list[dict] = Field(default_factory=list)


class TimingInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")
    analysis_s: float


class Timeline(BaseModel):
    """decision_timeline.json top level, exact keys per CONTRACT §10."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    schema_version: str = "1.0"
    tool: ToolInfo
    created_utc: str
    input: VideoInfo
    target: TargetInfo
    segment: SegmentInfo
    config_hash: str
    config: Config
    models: list[ModelEntry]
    analysis: AnalysisInfo
    face_tracks: list[FaceTrack]
    vad_segments: list[VadSegment]
    scene_cuts: list[float]
    decisions: list[Decision]
    speaker_segments: list[SpeakerSegment]
    switches: list[Switch]
    fallback_periods: list[FallbackPeriod]
    crops: list[Crop]
    invariant_violations: InvariantViolations
    warnings: list[str] = Field(default_factory=list)
    timing: TimingInfo


def save_timeline(timeline: Timeline, path: Union[Path, str]) -> None:
    """Write a Timeline to `path` as JSON (indent=1, forward-slash paths)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = timeline.model_dump(mode="json", by_alias=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1)


def load_timeline(path: Union[Path, str]) -> Timeline:
    """Read and validate a Timeline from `path` (raises on missing/invalid keys)."""
    p = Path(path)
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    return Timeline.model_validate(data)


__all__ = [
    "ToolInfo",
    "TargetInfo",
    "SegmentInfo",
    "ModelEntry",
    "AnalysisInfo",
    "TrackSample",
    "FaceTrack",
    "VadSegment",
    "Decision",
    "SpeakerSegment",
    "Switch",
    "FallbackPeriod",
    "Crop",
    "InvariantViolations",
    "TimingInfo",
    "Timeline",
    "save_timeline",
    "load_timeline",
]
