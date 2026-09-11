"""Unit tests for lips.window_features per CONTRACT §7."""

import numpy as np

from reframe.lips import window_features


def test_moving_mouth_gives_higher_lip_activity_than_constant():
    n = 20
    moving = [0.5 + 0.5 * ((i % 2) * 2 - 1) for i in range(n)]  # oscillates between 0 and 1
    constant = [0.3] * n
    energy = [0.0] * n

    lip_activity_moving, _ = window_features(moving, energy, k=n - 1, window_ticks=10)
    lip_activity_constant, _ = window_features(constant, energy, k=n - 1, window_ticks=10)

    assert lip_activity_moving > lip_activity_constant
    assert lip_activity_constant < 1e-9


def test_correlated_mouth_and_energy_give_high_av_corr():
    n = 20
    ramp = list(np.linspace(0.0, 1.0, n))

    _, av_corr = window_features(ramp, ramp, k=n - 1, window_ticks=n)
    assert av_corr > 0.99


def test_uncorrelated_mouth_and_energy_give_low_av_corr():
    mouth = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
    energy = [1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0]

    _, av_corr = window_features(mouth, energy, k=len(mouth) - 1, window_ticks=len(mouth))
    assert abs(av_corr) < 0.3


def test_constant_series_gives_zero_av_corr():
    """Both std < 1e-6 -> av_corr forced to 0 (Pearson is undefined for constant input)."""
    mouth = [0.5] * 10
    energy = [0.2] * 10

    lip_activity, av_corr = window_features(mouth, energy, k=9, window_ticks=10)
    assert lip_activity == 0.0
    assert av_corr == 0.0


def test_none_gaps_are_skipped():
    mouth = [None, None, 0.1, 0.9, None, 0.2, 0.8]
    energy = [0.1, 0.2, 0.1, 0.9, 0.3, 0.1, 0.8]

    lip_activity, _ = window_features(mouth, energy, k=6, window_ticks=7)
    assert lip_activity > 0.0


def test_all_none_window_gives_zeros():
    mouth = [None, None, None]
    energy = [0.1, 0.2, 0.3]

    lip_activity, av_corr = window_features(mouth, energy, k=2, window_ticks=3)
    assert lip_activity == 0.0
    assert av_corr == 0.0
