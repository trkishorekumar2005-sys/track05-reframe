"""Unit tests for the decision state machine per CONTRACT §8."""

from reframe.config import AudioConfig, Config, SpeakerConfig, TrackingConfig
from reframe.decide import build_fallback_periods, build_speaker_segments, decide


def _tick(t, visible_ids, scores, vad=0.0, is_cut=False, is_screen_content=False):
    return {
        "t": t,
        "visible_ids": visible_ids,
        "scores": scores,
        "vad": vad,
        "is_cut": is_cut,
        "is_screen_content": is_screen_content,
    }


def test_scene_cut_resets_to_fallback():
    """Rule 1: a scene cut always forces FALLBACK scene_cut_reset, even mid-lock."""
    ticks = [
        _tick(0.0, [1], {1: 0.9}),
        _tick(0.1, [1], {1: 0.9}, is_cut=True),
    ]
    decisions, switches = decide(ticks, Config(), has_audio=True)

    assert decisions[0]["mode"] == "speaker"
    assert decisions[1]["mode"] == "fallback"
    assert decisions[1]["reason"] == "scene_cut_reset"
    assert any(s["from"] == 1 and s["to"] == "FALLBACK" for s in switches)


def test_v_empty_grace_then_no_face():
    """Rule 2: with V completely empty, grace holds <= max_missed_s, then FALLBACK no_face."""
    cfg = Config(tracking=TrackingConfig(max_missed_s=0.2))
    ticks = [
        _tick(0.0, [1], {1: 0.9}),
        _tick(0.1, [], {}),
        _tick(0.2, [], {}),
        _tick(0.3, [], {}),
    ]
    decisions, _ = decide(ticks, cfg, has_audio=True)

    assert decisions[0]["mode"] == "speaker" and decisions[0]["track_id"] == 1
    assert decisions[1]["reason"] == "face_lost_grace" and decisions[1]["mode"] == "speaker"
    assert decisions[2]["reason"] == "face_lost_grace" and decisions[2]["mode"] == "speaker"
    assert decisions[3]["reason"] == "no_face" and decisions[3]["mode"] == "fallback"


def test_locked_face_lost_grace_then_face_lost_with_others_visible():
    """Rule 4(a): the LOCKED track itself vanishes (while others stay visible) -> face_lost.

    Distinct from rule 2 (V entirely empty -> no_face/screen_content): here another
    face remains visible the whole time, so V is never empty, but track 1 (locked)
    specifically disappears past max_missed_s.
    """
    cfg = Config(tracking=TrackingConfig(max_missed_s=0.2), speaker=SpeakerConfig(min_hold_s=0.0))
    ticks = [
        _tick(0.0, [1], {1: 0.9}),  # locks onto 1 (only face visible)
        _tick(0.1, [2], {2: 0.9}),  # 1 vanished, 2 appears; within grace
        _tick(0.2, [2], {2: 0.9}),  # still within grace (<=0.2)
        _tick(0.3, [2], {2: 0.9}),  # now > max_missed_s -> face_lost
    ]
    decisions, _ = decide(ticks, cfg, has_audio=True)

    assert decisions[0]["mode"] == "speaker" and decisions[0]["track_id"] == 1
    assert decisions[1]["reason"] == "face_lost_grace" and decisions[1]["track_id"] == 1
    assert decisions[2]["reason"] == "face_lost_grace" and decisions[2]["track_id"] == 1
    assert decisions[3]["reason"] == "face_lost" and decisions[3]["mode"] == "fallback"


def test_no_switch_before_min_hold():
    """Rule 4(b): a fresh lock is protected by min_hold_s even with a strong challenger."""
    cfg = Config(
        speaker=SpeakerConfig(min_hold_s=0.5, switch_margin=0.1, switch_confirm_s=0.05),
    )
    ticks = [_tick(0.0, [1], {1: 0.9})]
    ticks += [_tick(round(0.1 * i, 3), [1, 2], {1: 0.1, 2: 0.9}) for i in range(1, 5)]
    decisions, _ = decide(ticks, cfg, has_audio=True)

    for d in decisions[1:]:
        assert d["track_id"] == 1
        assert d["reason"] == "hold_time_active"


def test_switch_only_after_margin_held_for_confirm_s():
    """Rule 4(d): a sustained challenger margin only switches once held >= switch_confirm_s."""
    cfg = Config(
        speaker=SpeakerConfig(min_hold_s=0.0, switch_margin=0.1, switch_confirm_s=0.3),
    )
    ticks = [_tick(0.0, [1], {1: 0.9})]
    ticks += [_tick(round(0.1 * i, 3), [1, 2], {1: 0.1, 2: 0.9}) for i in range(1, 5)]
    decisions, _ = decide(ticks, cfg, has_audio=True)

    # Condition first true at t=0.1; needs 0.3s held -> satisfied at t=0.4.
    for d in decisions[1:4]:
        assert d["track_id"] == 1
    assert decisions[4]["track_id"] == 2
    assert decisions[4]["reason"] == "challenger_margin_sustained"


def test_challenger_burst_causes_no_switch():
    """A challenger margin sustained for only 0.3s (< switch_confirm_s=0.5) never switches."""
    cfg = Config(
        speaker=SpeakerConfig(min_hold_s=0.0, switch_margin=0.1, switch_confirm_s=0.5),
    )
    ticks = [_tick(0.0, [1], {1: 0.9})]
    # Challenger margin true from t=0.1 to t=0.4 (0.3s), then breaks at t=0.5.
    ticks += [_tick(round(0.1 * i, 3), [1, 2], {1: 0.1, 2: 0.9}) for i in range(1, 5)]
    ticks.append(_tick(0.5, [1, 2], {1: 0.9, 2: 0.1}))
    ticks += [_tick(round(0.1 * i, 3), [1, 2], {1: 0.9, 2: 0.1}) for i in range(6, 9)]
    decisions, switches = decide(ticks, cfg, has_audio=True)

    assert all(d["track_id"] == 1 for d in decisions)
    assert not any(s["to"] == 2 for s in switches)


def test_speech_no_confident_face_triggers_off_screen_voice():
    """Rule 4(c): sustained speech with a low-confidence speaker -> FALLBACK off_screen_voice."""
    cfg = Config(
        speaker=SpeakerConfig(min_hold_s=0.0, fallback_conf=0.2, fallback_enter_s=0.3),
        audio=AudioConfig(speech_threshold=0.5),
    )
    ticks = [_tick(0.0, [1], {1: 0.9})]
    ticks += [_tick(round(0.1 * i, 3), [1], {1: 0.1}, vad=0.9) for i in range(1, 5)]
    decisions, _ = decide(ticks, cfg, has_audio=True)

    for d in decisions[1:4]:
        assert d["mode"] == "speaker"
    assert decisions[4]["mode"] == "fallback"
    assert decisions[4]["reason"] == "off_screen_voice"


def test_silence_holds_locked_state():
    """Rule 4(e): with no challenger and no speech, stays LOCKED with reason silence_hold."""
    cfg = Config(speaker=SpeakerConfig(min_hold_s=0.0))
    ticks = [
        _tick(0.0, [1], {1: 0.9}),
        _tick(0.1, [1], {1: 0.9}, vad=0.0),
    ]
    decisions, _ = decide(ticks, cfg, has_audio=True)

    assert decisions[1]["mode"] == "speaker" and decisions[1]["track_id"] == 1
    assert decisions[1]["reason"] == "silence_hold"


def test_larger_face_zero_score_never_selected_by_av_scores():
    """Rule 5(c)/(d): under a non-largest_face strategy, a zero-score face is never locked."""
    cfg = Config(speaker=SpeakerConfig(strategy="av_heuristic", enter_conf=0.35))
    ticks = [_tick(round(0.1 * i, 3), [1], {1: 0.0}) for i in range(5)]
    decisions, _ = decide(ticks, cfg, has_audio=True)

    assert all(d["mode"] == "fallback" and d["track_id"] is None for d in decisions)


def test_no_audio_single_face_locks():
    """Rule 5(b): with no audio track and exactly one visible face, lock instantly."""
    cfg = Config(speaker=SpeakerConfig(strategy="av_heuristic"))
    ticks = [_tick(0.0, [1], {1: 0.0})]
    decisions, _ = decide(ticks, cfg, has_audio=False)

    assert decisions[0]["mode"] == "speaker"
    assert decisions[0]["track_id"] == 1
    assert decisions[0]["reason"] == "single_face_no_audio"


def test_largest_face_strategy_locks_instantly():
    """Rule 5(a): the largest_face strategy locks the best candidate with no confirm delay."""
    cfg = Config(speaker=SpeakerConfig(strategy="largest_face"))
    ticks = [_tick(0.0, [1], {1: 0.0})]  # score is irrelevant to this rule
    decisions, _ = decide(ticks, cfg, has_audio=True)

    assert decisions[0]["mode"] == "speaker"
    assert decisions[0]["track_id"] == 1
    assert decisions[0]["reason"] == "largest_face"


def test_tie_break_uses_lower_id():
    """Rule 3: argmax score ties break to the lower track id."""
    cfg = Config(speaker=SpeakerConfig(strategy="largest_face"))
    ticks = [_tick(0.0, [2, 1], {2: 0.5, 1: 0.5})]
    decisions, _ = decide(ticks, cfg, has_audio=True)

    assert decisions[0]["track_id"] == 1


def test_build_speaker_segments_groups_consecutive_ticks():
    decisions = [
        {"t": 0.0, "mode": "speaker", "track_id": 1, "confidence": 0.8, "reason": "x", "scores": {}},
        {"t": 0.1, "mode": "speaker", "track_id": 1, "confidence": 0.6, "reason": "y", "scores": {}},
        {"t": 0.2, "mode": "fallback", "track_id": None, "confidence": 0.0, "reason": "z", "scores": {}},
    ]
    segments = build_speaker_segments(decisions)
    assert len(segments) == 2
    assert segments[0]["start"] == 0.0 and segments[0]["end"] == 0.1
    assert segments[0]["track_id"] == 1
    assert segments[0]["mean_confidence"] == 0.7
    assert segments[1]["mode"] == "fallback"


def test_build_fallback_periods_merges_consecutive_fallback_ticks():
    decisions = [
        {"t": 0.0, "mode": "speaker", "track_id": 1, "confidence": 0.9, "reason": "a", "scores": {}},
        {"t": 0.1, "mode": "fallback", "track_id": None, "confidence": 0.0, "reason": "no_face", "scores": {}},
        {"t": 0.2, "mode": "fallback", "track_id": None, "confidence": 0.0, "reason": "no_face", "scores": {}},
        {"t": 0.3, "mode": "speaker", "track_id": 1, "confidence": 0.9, "reason": "b", "scores": {}},
    ]
    periods = build_fallback_periods(decisions)
    assert len(periods) == 1
    assert periods[0] == {"start": 0.1, "end": 0.2, "reason": "no_face"}
