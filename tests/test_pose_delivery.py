"""Task 4 tests, delivery.py.  ALL INPUTS ARE SYNTHETIC (seeded sigmoids / noise), not real pitcher data."""
import numpy as np

from pitchsim.pose.delivery import average_delivery, jitter, landmarks

HAND = 23  # right_hand


def _clip(fps, n, rel, width_s, rng=None, noise=0.0):
    """Synthetic clip: the right hand sweeps 1 m along x as a sigmoid whose speed peaks exactly at `rel`."""
    t = (np.arange(n) - rel) / fps
    j = np.zeros((n, 24, 3))
    j[:, HAND, 0] = 1.0 / (1.0 + np.exp(-t / width_s))
    j[:, 17, 1] = 0.3 * np.sin(2 * t)  # something else that moves
    if rng is not None:
        j += rng.normal(0, noise, j.shape)
    return j


def test_average_aligns_release():
    a = _clip(120, 240, 90, 0.05)
    b = _clip(90, 150, 60, 0.04)
    marks = [dict(leg_lift=30, foot_strike=60, release=90, fps=120.0),
             dict(leg_lift=15, foot_strike=40, release=60, fps=90.0)]
    j, ri = average_delivery([a, b], marks, pre_s=0.5)
    assert ri == round(0.5 * 60.0)
    speed = np.linalg.norm(np.gradient(j[:, HAND], axis=0), axis=1)
    assert abs(int(np.argmax(speed)) - ri) <= 1


def test_average_drops_short_clip():
    good = _clip(60, 120, 40, 0.05)
    short = _clip(60, 120, 18, 0.05)  # only 0.3 s before release
    m = dict(leg_lift=10, foot_strike=25, release=40, fps=60.0)
    ms = dict(leg_lift=2, foot_strike=10, release=18, fps=60.0)
    excluded = []
    j, ri = average_delivery([good, short, good], [m, ms, m], pre_s=0.5, excluded=excluded)
    assert [(i, r) for i, r in excluded] == [(1, "short")]


def test_smoothing_reduces_jitter():
    rng = np.random.default_rng(0)
    clips = [_clip(60, 120, 40, 0.1, rng, 0.003) for _ in range(3)]
    m = dict(leg_lift=10, foot_strike=25, release=40, fps=60.0)
    j, _ = average_delivery(clips, [m] * 3, pre_s=0.5)
    assert jitter(j, 60.0) < 0.5 * jitter(clips[0], 60.0)


def test_landmarks_order():
    f = 60
    t = np.arange(90) / f
    j = np.zeros((90, 24, 3))
    j[:, 4, 2] = np.exp(-((t - 0.5) / 0.15) ** 2)          # left knee peaks at frame 30 (z up)
    j[:, 7, 1] = -np.clip(t - 0.5, 0, 0.3) / 0.3 * 1.0     # left ankle strides, then stops at ~frame 48
    lm = landmarks(j, release=70, throws="R")
    assert lm["leg_lift"] == 30
    assert lm["leg_lift"] < lm["foot_strike"] < lm["release"] == 70
    assert 44 <= lm["foot_strike"] <= 52
