"""Unit tests for configuration loading, overrides, and hashing."""

import pytest
from pydantic import ValidationError

from reframe.config import Config, config_hash, load_config
from reframe.errors import InvalidInputError


def test_default_config_loads():
    """Default configuration loads with exact values from CONTRACT §17.

    speaker.strategy is the one deliberate deviation: tuned from the
    contract's literal "largest_face" starting point to "av_heuristic" once
    audio/lip features made it usable (§17 configs are explicitly "STARTING
    points, tuned only on dev clips").
    """
    cfg = load_config()
    assert isinstance(cfg, Config)

    # analysis
    assert cfg.analysis.sample_fps == 10
    assert cfg.analysis.analysis_width == 640
    assert cfg.analysis.detector == "mediapipe"
    assert cfg.analysis.min_face_conf == 0.5
    assert cfg.analysis.max_faces == 4

    # tracking
    assert cfg.tracking.iou_match == 0.3
    assert cfg.tracking.min_hits == 2
    assert cfg.tracking.max_missed_s == 0.7

    # audio
    assert cfg.audio.vad_aggressiveness == 2
    assert cfg.audio.speech_threshold == 0.5

    # speaker
    assert cfg.speaker.strategy == "av_heuristic"
    assert cfg.speaker.window_s == 0.8
    assert cfg.speaker.min_hold_s == 1.2
    assert cfg.speaker.switch_margin == 0.15
    assert cfg.speaker.switch_confirm_s == 0.4
    assert cfg.speaker.enter_conf == 0.35
    assert cfg.speaker.fallback_conf == 0.2
    assert cfg.speaker.fallback_enter_s == 0.6
    assert cfg.speaker.lip_activity_norm == 0.05

    # scene
    assert cfg.scene.enabled is True
    assert cfg.scene.cut_threshold == 0.5
    assert cfg.scene.screen_edge_density == 0.08

    # crop
    assert cfg.crop.aspect == "9:16"
    assert cfg.crop.headroom == 0.38
    assert cfg.crop.deadzone_frac == 0.04
    assert cfg.crop.max_speed_frac_s == 0.5
    assert cfg.crop.smoothing_alpha == 0.15
    assert cfg.crop.safety_margin_frac == 0.1

    # render
    assert cfg.render.crf == 20
    assert cfg.render.preset == "veryfast"
    assert cfg.render.fallback_style == "blur_pad"
    assert cfg.render.debug_width == 960

    # runtime
    assert cfg.runtime.seed == 1234
    assert cfg.runtime.cache_dir == ".cache"
    assert cfg.runtime.max_duration_s == 900.0
    assert cfg.runtime.cv_threads == 0


def test_unknown_key_rejected():
    """Unknown keys are rejected due to extra='forbid'."""
    with pytest.raises(ValidationError):
        load_config(overrides=["unknown_key=123"])

    with pytest.raises(ValidationError):
        load_config(overrides=["speaker.nonexistent_option=value"])

    with pytest.raises(ValidationError):
        load_config(overrides=["analysis.extra_property=42"])


def test_negative_speaker_min_hold_s_rejected():
    """Negative speaker.min_hold_s is rejected by validation bounds."""
    with pytest.raises(ValidationError):
        load_config(overrides=["speaker.min_hold_s=-1.0"])

    with pytest.raises(ValidationError):
        load_config(overrides=["speaker.min_hold_s=-0.01"])


def test_set_overrides_parse_int_float_str():
    """--set overrides correctly parse int, float, and string values."""
    # Test int override
    cfg_int = load_config(overrides=["render.crf=18"])
    assert cfg_int.render.crf == 18
    assert isinstance(cfg_int.render.crf, int)

    # Test float override
    cfg_float = load_config(overrides=["speaker.min_hold_s=2.5"])
    assert cfg_float.speaker.min_hold_s == 2.5
    assert isinstance(cfg_float.speaker.min_hold_s, float)

    # Test string override
    cfg_str = load_config(overrides=["crop.aspect=1:1"])
    assert cfg_str.crop.aspect == "1:1"
    assert isinstance(cfg_str.crop.aspect, str)

    # Test boolean override
    cfg_bool = load_config(overrides=["scene.enabled=false"])
    assert cfg_bool.scene.enabled is False
    assert isinstance(cfg_bool.scene.enabled, bool)

    # Test combination of multiple overrides
    cfg_multi = load_config(
        overrides=[
            "render.crf=25",
            "speaker.min_hold_s=3.75",
            "crop.aspect=1:1",
            "speaker.strategy=av_heuristic",
        ]
    )
    assert cfg_multi.render.crf == 25
    assert isinstance(cfg_multi.render.crf, int)
    assert cfg_multi.speaker.min_hold_s == 3.75
    assert isinstance(cfg_multi.speaker.min_hold_s, float)
    assert cfg_multi.crop.aspect == "1:1"
    assert isinstance(cfg_multi.crop.aspect, str)
    assert cfg_multi.speaker.strategy == "av_heuristic"


def test_config_hash_stability_and_change():
    """config_hash is deterministic (stable) and changes when a value changes."""
    cfg1 = load_config()
    cfg2 = load_config()

    h1 = config_hash(cfg1)
    h2 = config_hash(cfg2)

    # Stability: same config produces identical 16-hex hash
    assert h1 == h2
    assert len(h1) == 16
    assert isinstance(h1, str)
    int(h1, 16)  # Verifies it's a valid hex string

    # Hash changes when a value changes
    cfg_mod1 = load_config(overrides=["speaker.min_hold_s=2.0"])
    h_mod1 = config_hash(cfg_mod1)
    assert h_mod1 != h1
    assert len(h_mod1) == 16

    cfg_mod2 = load_config(overrides=["render.crf=21"])
    h_mod2 = config_hash(cfg_mod2)
    assert h_mod2 != h1
    assert h_mod2 != h_mod1


def test_invalid_config_file_not_found():
    """Loading from a non-existent file path raises InvalidInputError."""
    with pytest.raises(InvalidInputError) as exc_info:
        load_config(path="configs/non_existent_file.yaml")
    assert exc_info.value.code == "file_not_found"
    assert exc_info.value.exit_code == 2


def test_invalid_override_syntax():
    """Override without '=' raises InvalidInputError."""
    with pytest.raises(InvalidInputError) as exc_info:
        load_config(overrides=["invalid_override_format"])
    assert exc_info.value.code == "invalid_override"
    assert exc_info.value.exit_code == 2
