"""Configuration models and loading per CONTRACT §2, §16, §17."""

import hashlib
import json
from pathlib import Path
from typing import Any, Optional, Union
import yaml
from pydantic import BaseModel, ConfigDict, Field

from reframe.errors import InvalidInputError


class AnalysisConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sample_fps: int = Field(10, gt=0, le=120)
    analysis_width: int = Field(640, ge=64, le=7680)
    detector: str = "mediapipe"
    min_face_conf: float = Field(0.25, ge=0.0, le=1.0)
    max_faces: int = Field(4, ge=1, le=100)


class TrackingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    iou_match: float = Field(0.3, ge=0.0, le=1.0)
    min_hits: int = Field(2, ge=1)
    max_missed_s: float = Field(0.7, ge=0.0)


class AudioConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    vad_aggressiveness: int = Field(2, ge=0, le=3)
    speech_threshold: float = Field(0.5, ge=0.0, le=1.0)


class SpeakerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    strategy: str = "largest_face"
    window_s: float = Field(0.8, ge=0.0)
    min_hold_s: float = Field(1.2, ge=0.0)
    switch_margin: float = Field(0.15, ge=0.0, le=1.0)
    switch_confirm_s: float = Field(0.4, ge=0.0)
    enter_conf: float = Field(0.35, ge=0.0, le=1.0)
    fallback_conf: float = Field(0.2, ge=0.0, le=1.0)
    fallback_enter_s: float = Field(0.6, ge=0.0)
    lip_activity_norm: float = Field(0.05, ge=0.0)


class SceneConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = True
    cut_threshold: float = Field(0.5, ge=0.0, le=1.0)
    screen_edge_density: float = Field(0.08, ge=0.0, le=1.0)


class CropConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    aspect: str = "9:16"
    headroom: float = Field(0.38, ge=0.0, le=1.0)
    deadzone_frac: float = Field(0.04, ge=0.0, le=1.0)
    max_speed_frac_s: float = Field(0.5, ge=0.0)
    smoothing_alpha: float = Field(0.15, ge=0.0, le=1.0)
    safety_margin_frac: float = Field(0.1, ge=0.0, le=0.5)


class RenderConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    crf: int = Field(20, ge=0, le=51)
    preset: str = "veryfast"
    fallback_style: str = "blur_pad"
    debug_width: int = Field(960, ge=64)


class RuntimeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    seed: int = Field(1234, ge=0)
    cache_dir: str = ".cache"
    max_duration_s: float = Field(900.0, gt=0.0)
    cv_threads: int = Field(0, ge=0)


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    speaker: SpeakerConfig = Field(default_factory=SpeakerConfig)
    scene: SceneConfig = Field(default_factory=SceneConfig)
    crop: CropConfig = Field(default_factory=CropConfig)
    render: RenderConfig = Field(default_factory=RenderConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)


def _parse_scalar(val: str) -> Any:
    """Parse override string value into bool, int, float, or str."""
    val_stripped = val.strip()
    if val_stripped.lower() == "true":
        return True
    if val_stripped.lower() == "false":
        return False
    if val_stripped.lower() in ("none", "null"):
        return None
    if (val_stripped.startswith('"') and val_stripped.endswith('"')) or (
        val_stripped.startswith("'") and val_stripped.endswith("'")
    ):
        return val_stripped[1:-1]
    try:
        if ":" not in val_stripped and "." not in val_stripped:
            return int(val_stripped)
    except ValueError:
        pass
    try:
        if ":" not in val_stripped:
            return float(val_stripped)
    except ValueError:
        pass
    return val_stripped


def _apply_override(cfg_dict: dict[str, Any], key: str, value_str: str) -> None:
    """Apply a dotted key=value override to a nested dictionary."""
    parts = key.split(".")
    curr = cfg_dict
    for part in parts[:-1]:
        if part not in curr or not isinstance(curr[part], dict):
            curr[part] = {}
        curr = curr[part]
    curr[parts[-1]] = _parse_scalar(value_str)


def load_config(
    path: Union[Path, str, None] = None,
    overrides: Optional[list[str]] = None,
) -> Config:
    """Load configuration from YAML and apply overrides per CONTRACT §2."""
    if path is None:
        default_path = Path("configs/default.yaml")
        if not default_path.is_file():
            pkg_root = Path(__file__).resolve().parent.parent.parent
            default_path = pkg_root / "configs" / "default.yaml"
        if default_path.is_file():
            path = default_path

    raw_dict: dict[str, Any] = {}
    if path is not None:
        p = Path(path)
        if not p.is_file():
            raise InvalidInputError(
                f"Config file not found: {path}", code="file_not_found"
            )
        try:
            with open(p, "r", encoding="utf-8") as f:
                content = yaml.safe_load(f)
                if content is not None:
                    if not isinstance(content, dict):
                        raise InvalidInputError(
                            f"Invalid YAML root structure in {path}",
                            code="invalid_config",
                        )
                    raw_dict = content
        except yaml.YAMLError as exc:
            raise InvalidInputError(
                f"Failed to parse YAML from {path}: {exc}", code="invalid_config"
            ) from exc

    if overrides:
        for item in overrides:
            if "=" not in item:
                raise InvalidInputError(
                    f"Invalid override format '{item}', expected 'key=value'",
                    code="invalid_override",
                )
            k, v = item.split("=", 1)
            _apply_override(raw_dict, k.strip(), v.strip())

    return Config.model_validate(raw_dict)


def config_hash(cfg: Config) -> str:
    """Compute sha256 of canonical sorted JSON, first 16 hex characters per CONTRACT §2."""
    raw_dict = cfg.model_dump()
    canonical_json = json.dumps(raw_dict, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()[:16]


__all__ = [
    "Config",
    "AnalysisConfig",
    "TrackingConfig",
    "AudioConfig",
    "SpeakerConfig",
    "SceneConfig",
    "CropConfig",
    "RenderConfig",
    "RuntimeConfig",
    "load_config",
    "config_hash",
]
