"""Unit tests for IoUTracker and box_at per CONTRACT §6."""

import pytest

from reframe.config import Config, TrackingConfig
from reframe.detect import Detection
from reframe.track import IoUTracker, box_at


def _cfg(max_missed_s: float = 0.25) -> Config:
    return Config(tracking=TrackingConfig(iou_match=0.3, min_hits=2, max_missed_s=max_missed_s))


def test_one_moving_face_keeps_one_id():
    """A single face moving slightly across ticks stays under one track id."""
    tracker = IoUTracker(_cfg())
    for i in range(10):
        t = i * 0.1
        x = 100.0 + i * 2.0  # small consistent motion, still high IoU vs previous box
        det = Detection(bbox=[x, 100.0, 200.0, 200.0], conf=0.9)
        tracker.update(t, [det])

    tracks = tracker.finalize()
    assert len(tracks) == 1
    assert len(tracks[0]["samples"]) == 10
    assert tracks[0]["first_t"] == 0.0
    assert tracks[0]["last_t"] == pytest.approx(0.9)


def test_two_faces_keep_separate_ids():
    """Two well-separated faces produce two distinct confirmed tracks."""
    tracker = IoUTracker(_cfg())
    for i in range(10):
        t = i * 0.1
        det_a = Detection(bbox=[0.0, 0.0, 100.0, 100.0], conf=0.9)
        det_b = Detection(bbox=[500.0, 500.0, 100.0, 100.0], conf=0.9)
        tracker.update(t, [det_a, det_b])

    tracks = tracker.finalize()
    assert len(tracks) == 2
    ids = {tr["track_id"] for tr in tracks}
    assert len(ids) == 2
    for tr in tracks:
        assert len(tr["samples"]) == 10


def test_disappearance_ends_track():
    """A face missing for longer than max_missed_s ends its track."""
    tracker = IoUTracker(_cfg(max_missed_s=0.25))

    # Present for 3 ticks (confirmed since min_hits=2).
    for i in range(3):
        t = i * 0.1
        tracker.update(t, [Detection(bbox=[100.0, 100.0, 200.0, 200.0], conf=0.9)])

    # Gone for longer than max_missed_s (0.25s).
    for i in range(3, 8):
        tracker.update(i * 0.1, [])

    tracks = tracker.finalize()
    assert len(tracks) == 1
    assert tracks[0]["last_t"] == pytest.approx(0.2)


def test_reentry_gets_new_id():
    """A face reappearing after a long gap gets a new track id, not the old one."""
    tracker = IoUTracker(_cfg(max_missed_s=0.25))

    for i in range(3):
        tracker.update(i * 0.1, [Detection(bbox=[100.0, 100.0, 200.0, 200.0], conf=0.9)])

    for i in range(3, 8):
        tracker.update(i * 0.1, [])

    # Re-entry at the same location well past max_missed_s.
    for i in range(8, 11):
        tracker.update(i * 0.1, [Detection(bbox=[100.0, 100.0, 200.0, 200.0], conf=0.9)])

    tracks = tracker.finalize()
    assert len(tracks) == 2
    ids = sorted(tr["track_id"] for tr in tracks)
    assert ids[0] != ids[1]
    assert ids[1] > ids[0]  # the re-entered track is a fresh, later id


def test_single_sample_flicker_dropped():
    """A one-off detection that's never matched again (hits < min_hits) is dropped."""
    tracker = IoUTracker(_cfg(max_missed_s=0.25))

    for i in range(12):
        t = i * 0.1
        dets = [Detection(bbox=[100.0, 100.0, 200.0, 200.0], conf=0.9)]
        if i == 2:
            dets.append(Detection(bbox=[900.0, 900.0, 50.0, 50.0], conf=0.4))  # isolated one-off
        tracker.update(t, dets)

    tracks = tracker.finalize()
    assert len(tracks) == 1
    assert tracks[0]["samples"][0]["bbox"][0] == 100.0


def test_box_at_interpolates_between_samples():
    """box_at() linearly interpolates bbox position between two adjacent samples."""
    tracker = IoUTracker(_cfg())
    tracker.update(0.0, [Detection(bbox=[0.0, 0.0, 100.0, 100.0], conf=0.9)])
    tracker.update(0.1, [Detection(bbox=[10.0, 0.0, 100.0, 100.0], conf=0.9)])
    tracks = tracker.finalize()
    track = tracks[0]

    assert box_at(track, 0.0) == [0.0, 0.0, 100.0, 100.0]
    assert box_at(track, 0.1) == [10.0, 0.0, 100.0, 100.0]
    mid = box_at(track, 0.05)
    assert mid is not None
    assert mid[0] == pytest.approx(5.0)


def test_box_at_none_outside_track_and_inside_gap():
    """box_at() returns None before/after the track and inside a missed-detection gap."""
    tracker = IoUTracker(_cfg(max_missed_s=1.0))  # generous, so the track survives the gap
    tracker.update(0.0, [Detection(bbox=[0.0, 0.0, 100.0, 100.0], conf=0.9)])
    tracker.update(0.1, [Detection(bbox=[0.0, 0.0, 100.0, 100.0], conf=0.9)])
    tracker.update(0.2, [])  # missed tick -> gap
    tracker.update(0.5, [Detection(bbox=[0.0, 0.0, 100.0, 100.0], conf=0.9)])
    tracks = tracker.finalize()
    track = tracks[0]

    assert box_at(track, -0.1) is None  # before track starts
    assert box_at(track, 0.6) is None  # after track ends
    assert box_at(track, 0.3) is None  # inside the gap between t=0.1 and t=0.5
