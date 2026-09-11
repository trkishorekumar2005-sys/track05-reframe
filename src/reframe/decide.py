"""Decision state machine per CONTRACT §8.

`decide()` is a pure function of a list of tick inputs; all state (locked
track, timers) is local to the call, threaded through the tick loop. Ticks
must be in strictly increasing `t` order at a fixed dt = 1/sample_fps.

Each tick input dict has:
    t: float                      # absolute seconds
    visible_ids: list[int]        # V
    scores: dict[int, float]      # s[track_id], visible tracks only
    vad: float                    # continuous vad[k] in [0, 1]
    is_cut: bool                  # scene cut at this tick
    is_screen_content: bool       # scene says this looks like screen content
"""

from __future__ import annotations

from typing import Optional

from reframe.config import Config

_NEG_INF = float("-inf")


def _update_accumulator(since: Optional[float], cond: bool, t: float) -> Optional[float]:
    """Start/keep/reset a "condition held continuously since" timer for one tick."""
    if not cond:
        return None
    return since if since is not None else t


def _argmax_score(scores: dict[int, float], ids: list[int]) -> Optional[int]:
    """argmax s over `ids` (tie -> lower id); None if `ids` is empty."""
    best_id: Optional[int] = None
    best_score = _NEG_INF
    for tid in sorted(ids):
        s = scores.get(tid, 0.0)
        if s > best_score:
            best_score = s
            best_id = tid
    return best_id


class _State:
    __slots__ = (
        "mode",
        "locked_id",
        "locked_since",
        "challenger_since",
        "offscreen_since",
        "fallback_enter_since",
        "last_visible_t",
    )

    def __init__(self) -> None:
        self.mode = "FALLBACK"
        self.locked_id: Optional[int] = None
        self.locked_since: Optional[float] = None
        self.challenger_since: Optional[float] = None
        self.offscreen_since: Optional[float] = None
        self.fallback_enter_since: Optional[float] = None
        self.last_visible_t: dict[int, float] = {}

    def _enter_fallback(self, reason: str) -> str:
        self.mode = "FALLBACK"
        self.locked_id = None
        self.locked_since = None
        self.challenger_since = None
        self.offscreen_since = None
        self.fallback_enter_since = None
        return reason

    def _enter_locked(self, track_id: int, t: float, reason: str) -> str:
        self.mode = "LOCKED"
        self.locked_id = track_id
        self.locked_since = t
        self.challenger_since = None
        self.offscreen_since = None
        self.fallback_enter_since = None
        return reason


def _decide_one_tick(state: _State, tick: dict, cfg: Config, has_audio: bool) -> str:
    """Apply CONTRACT §8's rules for a single tick; mutates `state`; returns the reason."""
    t = float(tick["t"])
    visible_ids: list[int] = tick["visible_ids"]
    scores: dict[int, float] = tick["scores"]
    vad = float(tick["vad"])
    is_cut = bool(tick["is_cut"])
    is_screen_content = bool(tick["is_screen_content"])

    for vid in visible_ids:
        state.last_visible_t[vid] = t

    speech = vad >= cfg.audio.speech_threshold

    # Rule 1: scene cut always wins, resets everything.
    if is_cut:
        return state._enter_fallback("scene_cut_reset")

    V = visible_ids

    # Rule 2: no visible faces at all.
    if not V:
        if state.mode == "LOCKED" and state.locked_id is not None:
            missed = t - state.last_visible_t.get(state.locked_id, _NEG_INF)
            if missed <= cfg.tracking.max_missed_s:
                return "face_lost_grace"  # stay LOCKED, no change
            return state._enter_fallback("screen_content" if is_screen_content else "no_face")
        return state._enter_fallback("screen_content" if is_screen_content else "no_face")

    # Rule 3: best visible candidate this tick.
    best = _argmax_score(scores, V)
    assert best is not None

    if state.mode == "LOCKED":
        a = state.locked_id
        assert a is not None

        if a not in V:
            # Rule 4(a)
            missed = t - state.last_visible_t.get(a, _NEG_INF)
            if missed > cfg.tracking.max_missed_s:
                return state._enter_fallback("face_lost")
            return "face_lost_grace"

        s_a = scores.get(a, 0.0)
        s_best = scores.get(best, 0.0)

        # Maintain the two condition-accumulators every tick (regardless of
        # whether a higher-priority rule below currently blocks acting on them).
        offscreen_cond = speech and (s_best < cfg.speaker.fallback_conf)
        state.offscreen_since = _update_accumulator(state.offscreen_since, offscreen_cond, t)

        challenger_cond = (best != a) and (s_best - s_a >= cfg.speaker.switch_margin)
        state.challenger_since = _update_accumulator(state.challenger_since, challenger_cond, t)

        # Rule 4(b)
        if t - state.locked_since < cfg.speaker.min_hold_s:
            return "hold_time_active"

        # Rule 4(c)
        if offscreen_cond and (t - state.offscreen_since) >= cfg.speaker.fallback_enter_s:
            return state._enter_fallback("off_screen_voice")

        # Rule 4(d)
        if challenger_cond and (t - state.challenger_since) >= cfg.speaker.switch_confirm_s:
            return state._enter_locked(best, t, "challenger_margin_sustained")

        # Rule 4(e)
        if not speech:
            return "silence_hold"

        # Rule 4(f)
        return "audio_visual_match"

    # FALLBACK state, V non-empty.
    # Rule 5(a)
    if cfg.speaker.strategy == "largest_face":
        return state._enter_locked(best, t, "largest_face")

    # Rule 5(b)
    if not has_audio and len(V) == 1:
        return state._enter_locked(best, t, "single_face_no_audio")

    # Rule 5(c)
    s_best = scores.get(best, 0.0)
    enter_cond = s_best >= cfg.speaker.enter_conf
    state.fallback_enter_since = _update_accumulator(state.fallback_enter_since, enter_cond, t)
    if enter_cond and (t - state.fallback_enter_since) >= cfg.speaker.switch_confirm_s:
        return state._enter_locked(best, t, "audio_visual_match")

    # Rule 5(d)
    return "off_screen_voice" if speech else "low_confidence"


def decide(ticks: list[dict], cfg: Config, has_audio: bool) -> tuple[list[dict], list[dict]]:
    """Run the CONTRACT §8 state machine over `ticks`. Returns (decisions, switches)."""
    state = _State()
    decisions: list[dict] = []
    switches: list[dict] = []
    prev_target: object = "FALLBACK"

    for tick in ticks:
        reason = _decide_one_tick(state, tick, cfg, has_audio)
        scores = dict(tick["scores"])

        if state.mode == "LOCKED":
            track_id = state.locked_id
            confidence = scores.get(track_id, 0.0)
            decision = {
                "t": tick["t"],
                "mode": "speaker",
                "track_id": track_id,
                "confidence": confidence,
                "reason": reason,
                "scores": scores,
            }
            target: object = track_id
        else:
            confidence = max(scores.values()) if scores else 0.0
            decision = {
                "t": tick["t"],
                "mode": "fallback",
                "track_id": None,
                "confidence": confidence,
                "reason": reason,
                "scores": scores,
            }
            target = "FALLBACK"

        decisions.append(decision)

        if target != prev_target:
            switches.append(
                {
                    "t": tick["t"],
                    "from": prev_target,
                    "to": target,
                    "reason": reason,
                    "confidence": confidence,
                }
            )
        prev_target = target

    return decisions, switches


def build_speaker_segments(decisions: list[dict]) -> list[dict]:
    """Merge consecutive ticks sharing (mode, track_id) into segments.

    Assumption: a segment's `end` is its last tick's own time (not
    extended to the next segment's start).
    """
    segments: list[dict] = []
    if not decisions:
        return segments

    start_idx = 0
    n = len(decisions)
    for i in range(1, n + 1):
        if i == n or (decisions[i]["mode"], decisions[i]["track_id"]) != (
            decisions[start_idx]["mode"],
            decisions[start_idx]["track_id"],
        ):
            chunk = decisions[start_idx:i]
            mean_conf = sum(d["confidence"] for d in chunk) / len(chunk)
            segments.append(
                {
                    "start": chunk[0]["t"],
                    "end": chunk[-1]["t"],
                    "mode": chunk[0]["mode"],
                    "track_id": chunk[0]["track_id"],
                    "mean_confidence": round(mean_conf, 3),
                    "reason": chunk[0]["reason"],
                }
            )
            start_idx = i
    return segments


def build_fallback_periods(decisions: list[dict]) -> list[dict]:
    """Merge consecutive fallback-mode ticks into periods.

    Assumption: a period's `reason` is the reason of its first tick.
    """
    periods: list[dict] = []
    start_idx: Optional[int] = None

    for i, d in enumerate(decisions):
        if d["mode"] == "fallback":
            if start_idx is None:
                start_idx = i
        elif start_idx is not None:
            periods.append(
                {
                    "start": decisions[start_idx]["t"],
                    "end": decisions[i - 1]["t"],
                    "reason": decisions[start_idx]["reason"],
                }
            )
            start_idx = None

    if start_idx is not None:
        periods.append(
            {
                "start": decisions[start_idx]["t"],
                "end": decisions[-1]["t"],
                "reason": decisions[start_idx]["reason"],
            }
        )
    return periods


__all__ = ["decide", "build_speaker_segments", "build_fallback_periods"]
