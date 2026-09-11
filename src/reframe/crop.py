"""Crop planning per CONTRACT §9."""

from __future__ import annotations

import math
from typing import Optional

from reframe.config import Config
from reframe.track import box_at


def _parse_aspect(aspect: str) -> tuple[float, float]:
    aw_str, ah_str = aspect.split(":")
    return float(aw_str), float(ah_str)


def _even_floor(x: float) -> int:
    """Largest even integer <= x."""
    return (math.floor(x) // 2) * 2


def crop_size(W: int, H: int, aspect: str) -> tuple[int, int]:
    """Crop pixel size per CONTRACT §9 (never upscales, dimensions always even)."""
    aw, ah = _parse_aspect(aspect)
    r = aw / ah
    if W / H >= r:
        h = _even_floor(H)
        w = _even_floor(h * r)
    else:
        w = _even_floor(W)
        h = _even_floor(w / r)
    return w, h


def clamp_rect(
    rect: tuple[float, float, float, float], W: int, H: int
) -> tuple[float, float, float, float]:
    """Clamp a (x, y, w, h) rect to lie fully within [0, W] x [0, H]."""
    x, y, w, h = rect
    w = min(w, float(W))
    h = min(h, float(H))
    x = max(0.0, min(x, W - w))
    y = max(0.0, min(y, H - h))
    return (x, y, w, h)


def target_rect(
    face_box: list[float], W: int, H: int, w: int, h: int, headroom: float
) -> tuple[float, float, float, float]:
    """Target crop rect centred (horizontally) on the face, with vertical headroom."""
    fx, fy, fw, fh = face_box
    cx = fx + fw / 2.0
    cy = fy + fh / 2.0
    x = cx - w / 2.0
    y = cy - headroom * h
    return clamp_rect((x, y, float(w), float(h)), W, H)


def _smooth_axis(current: float, target: float, deadzone: float, max_step: float, alpha: float) -> float:
    d = target - current
    if abs(d) < deadzone:
        return current
    step = alpha * d
    if step > max_step:
        step = max_step
    elif step < -max_step:
        step = -max_step
    return current + step


class Smoother:
    """Per-axis crop-position smoother per CONTRACT §9.

    SNAP is decided by the caller (mode change, track change, or scene cut)
    and passed in as `snap`. Otherwise each axis moves by
    `crop.smoothing_alpha * distance`, capped at `crop.max_speed_frac_s * W / fps`,
    and holds still if the distance is under `crop.deadzone_frac * W`. The
    safety rule (face centre must stay inside the crop's central margin) is
    applied afterwards and can exceed the speed cap.
    """

    def __init__(self, cfg: Config, w: int, h: int, W: int, fps: float):
        self._w = w
        self._h = h
        self._deadzone = cfg.crop.deadzone_frac * W
        self._max_step = cfg.crop.max_speed_frac_s * W / fps
        self._alpha = cfg.crop.smoothing_alpha
        self._safety_margin_frac = cfg.crop.safety_margin_frac
        self._current: Optional[tuple[float, float]] = None

    def step(
        self,
        target_x: float,
        target_y: float,
        snap: bool,
        face_box: Optional[list[float]] = None,
    ) -> tuple[float, float]:
        if snap or self._current is None:
            new_x, new_y = target_x, target_y
        else:
            cur_x, cur_y = self._current
            new_x = _smooth_axis(cur_x, target_x, self._deadzone, self._max_step, self._alpha)
            new_y = _smooth_axis(cur_y, target_y, self._deadzone, self._max_step, self._alpha)

        if face_box is not None:
            new_x, new_y = self._apply_safety(new_x, new_y, face_box)

        self._current = (new_x, new_y)
        return new_x, new_y

    def _apply_safety(self, x: float, y: float, face_box: list[float]) -> tuple[float, float]:
        fx, fy, fw, fh = face_box
        fcx = fx + fw / 2.0
        fcy = fy + fh / 2.0
        margin_x = self._safety_margin_frac * self._w
        margin_y = self._safety_margin_frac * self._h
        lo_x, hi_x = x + margin_x, x + self._w - margin_x
        lo_y, hi_y = y + margin_y, y + self._h - margin_y

        if fcx < lo_x:
            x += fcx - lo_x
        elif fcx > hi_x:
            x += fcx - hi_x

        if fcy < lo_y:
            y += fcy - lo_y
        elif fcy > hi_y:
            y += fcy - hi_y

        return x, y


def check_crop_invariants(
    rect: tuple[float, float, float, float],
    W: int,
    H: int,
    aspect: str,
    mode: str,
    face_box: Optional[list[float]] = None,
) -> list[str]:
    """Sanity-check a final (int) crop rect per CONTRACT §9. Never auto-fixes."""
    violations: list[str] = []
    x, y, w, h = rect

    if x < 0 or y < 0 or x + w > W or y + h > H:
        violations.append("out_of_frame")

    if w % 2 != 0 or h % 2 != 0:
        violations.append("odd_dims")

    if mode == "speaker":
        aw, ah = _parse_aspect(aspect)
        r = aw / ah
        if h == 0 or abs(w / h - r) > 0.01:
            violations.append("wrong_aspect")
        if face_box is not None:
            fx, fy, fw, fh = face_box
            fcx = fx + fw / 2.0
            fcy = fy + fh / 2.0
            if not (x <= fcx <= x + w and y <= fcy <= y + h):
                violations.append("face_centre_outside")

    if mode == "fallback":
        if not (x == 0 and y == 0 and w == W and h == H):
            violations.append("bad_fallback_rect")

    return violations


def _tick_index(t_f: float, start_s: float, sample_fps: float, n_ticks: int) -> int:
    idx = math.floor((t_f - start_s) * sample_fps)
    return max(0, min(idx, n_ticks - 1))


def plan_crops(
    decisions: list[dict],
    tracks: list[dict],
    W: int,
    H: int,
    fps: float,
    start_s: float,
    n_frames: int,
    sample_fps: float,
    scene_cuts: list[float],
    cfg: Config,
) -> tuple[list[dict], list[dict]]:
    """Plan per-output-frame crop rects per CONTRACT §9.

    Returns (crops, violations) where crops is
    [{f, t, x, y, w, h, mode}] and violations is
    [{f, t, codes}] for frames with a non-empty check_crop_invariants result.

    If a decision names a speaker track whose interpolated box is
    unavailable at that instant (track ended or inside a tracking gap),
    the frame degrades to a full-frame fallback rect rather than fabricating
    a position.
    """
    if not decisions or n_frames <= 0:
        return [], []

    aspect = cfg.crop.aspect
    w, h = crop_size(W, H, aspect)
    tracks_by_id = {tr["track_id"]: tr for tr in tracks}

    scene_cut_ticks = {round((ct - start_s) * sample_fps) for ct in scene_cuts}

    smoother = Smoother(cfg, w, h, W, fps)
    crops: list[dict] = []
    violations: list[dict] = []

    last_mode: Optional[str] = None
    last_track_id: Optional[int] = None
    n_ticks = len(decisions)

    for f in range(n_frames):
        t_f = start_s + f / fps
        tick_idx = _tick_index(t_f, start_s, sample_fps, n_ticks)
        decision = decisions[tick_idx]
        mode = decision.get("mode", "fallback")
        track_id = decision.get("track_id")

        scene_cut = tick_idx in scene_cut_ticks
        snap = scene_cut or mode != last_mode or track_id != last_track_id

        face_box = None
        if mode == "speaker" and track_id is not None:
            track = tracks_by_id.get(track_id)
            face_box = box_at(track, t_f) if track is not None else None

        if face_box is not None:
            tx, ty, tw, th = target_rect(face_box, W, H, w, h, cfg.crop.headroom)
            x, y = smoother.step(tx, ty, snap, face_box=face_box)
            x, y, wc, hc = clamp_rect((x, y, float(w), float(h)), W, H)
            frame_mode = "speaker"
        else:
            x, y, wc, hc = 0.0, 0.0, float(W), float(H)
            frame_mode = "fallback"

        rect_int = (int(round(x)), int(round(y)), int(round(wc)), int(round(hc)))
        codes = check_crop_invariants(rect_int, W, H, aspect, frame_mode, face_box=face_box)
        if codes:
            violations.append({"f": f, "t": round(t_f, 3), "codes": codes})

        crops.append(
            {
                "f": f,
                "t": round(t_f, 3),
                "x": rect_int[0],
                "y": rect_int[1],
                "w": rect_int[2],
                "h": rect_int[3],
                "mode": frame_mode,
            }
        )

        last_mode = frame_mode
        last_track_id = track_id if frame_mode == "speaker" else None

    return crops, violations


__all__ = [
    "crop_size",
    "clamp_rect",
    "target_rect",
    "Smoother",
    "check_crop_invariants",
    "plan_crops",
]
