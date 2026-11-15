"""Task 4 tests, delivery.py.  ALL INPUTS ARE SYNTHETIC (seeded sigmoids / noise), not real pitcher data."""
import numpy as np

from pitchsim.pose.delivery import average_delivery, ground_root, jitter, landmarks

HAND = 23  # right_hand


def _clip(fps, n, rel, width_s, rng=None, noise=0.0, fs=None):
    """Synthetic clip: the right hand sweeps 1 m along x as a sigmoid whose speed peaks exactly at `rel`."""
    t = (np.arange(n) - rel) / fps
    j = np.zeros((n, 24, 3))
    j[:, HAND, 0] = 1.0 / (1.0 + np.exp(-t / width_s))
    j[:, 17, 1] = 0.3 * np.sin(2 * t)  # something else that moves
    if fs is not None:  # synthetic feature that peaks at foot_strike (pelvis y)
        j[:, 0, 1] = np.exp(-(((np.arange(n) - fs) / fps) / 0.05) ** 2 / 2)
    if rng is not None:
        j += rng.normal(0, noise, j.shape)
    return j


def test_average_aligns_release():
    a = _clip(120, 240, 90, 0.05, fs=60)
    b = _clip(90, 150, 60, 0.04, fs=40)
    marks = [dict(leg_lift=30, foot_strike=60, release=90, fps=120.0),
             dict(leg_lift=15, foot_strike=40, release=60, fps=90.0)]
    j, ri = average_delivery([a, b], marks, pre_s=0.5)
    assert ri == round(0.5 * 60.0)
    speed = np.linalg.norm(np.gradient(j[:, HAND], axis=0), axis=1)
    assert abs(int(np.argmax(speed)) - ri) <= 1
    # foot_strike template time = median(-0.25 s, -0.222 s): a broken warp would misplace this bump
    assert abs(int(np.argmax(j[:, 0, 1])) - (ri + (-0.25 - 2 / 9) / 2 * 60)) <= 1.2


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


def test_average_drops_clip_short_after_warp():
    """Passes the raw pre_s check but its early leg lift needs frames before 0 once warped."""
    early = _clip(60, 120, 40, 0.05)  # 0.667 s before release >= pre_s
    ok = _clip(60, 120, 40, 0.05)
    m_early = dict(leg_lift=1, foot_strike=15, release=40, fps=60.0)  # leg lift 0.65 s before release
    m_ok = dict(leg_lift=22, foot_strike=30, release=40, fps=60.0)    # 0.3 s before release
    excluded = []
    average_delivery([early, ok, ok], [m_early, m_ok, m_ok], pre_s=0.5, excluded=excluded)
    assert excluded == [(0, "short")]


def test_average_rejects_tiny_output():
    c = _clip(60, 120, 40, 0.05)
    m = dict(leg_lift=10, foot_strike=25, release=40, fps=60.0)
    try:
        average_delivery([c], [m], pre_s=0.0, post_s=0.0)
    except ValueError as e:
        assert "Savitzky" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_ground_root_removes_root_depth_noise():
    """SYNTHETIC: a rigid body whose root drifts wildly in depth (like the real lifts) while the pivot foot
    is planted until foot strike and the lead foot after it.  Grounding must pin both feet."""
    rng = np.random.default_rng(0)
    n, fs = 80, 40
    j = np.zeros((n, 24, 3))
    j[:, 11] = [0.0, 0.0, 0.0]       # right toe (pivot for a righty) at origin ...
    j[:, 10] = [0.2, 0.0, 0.0]       # left toe (lead) a stride away, relative to the body
    j[fs:, 10, 2] = 0.0
    root = np.zeros((n, 3))
    root[:, 2] = rng.normal(0, 5.0, n)             # +-5 m depth noise on the whole body
    jn = j + root[:, None, :]
    g = ground_root(jn, "R", fs)
    assert np.allclose(g[: fs + 1, 11], g[0, 11], atol=0.05)          # pivot pinned until strike
    assert np.allclose(g[fs:, 10], g[fs, 10], atol=0.05)              # lead pinned afterwards


def test_time_warp_is_smooth_monotone_and_hits_the_knots():
    """SYNTHETIC knots.  A piecewise-linear warp kinks at each landmark (the warped clip's speed jumps
    there); the smooth warp passes through the same knots with continuous slope."""
    from pitchsim.pose.delivery import time_warp
    tmpl, clip = np.array([-2.5, -1.0, -0.2, 0.0]), np.array([-2.7, -1.3, -0.15, 0.0])
    w = time_warp(tmpl, clip)
    assert np.allclose(w(tmpl), clip)
    x = np.linspace(-3, 0.6, 2001)
    assert np.all(np.diff(w(x)) > 0)
    h = 1e-5
    for k in tmpl:
        left, right = (w(k) - w(k - h)) / h, (w(k + h) - w(k)) / h
        assert abs(left - right) < 1e-3, (k, left, right)
