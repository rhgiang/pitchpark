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


def _rocker_clip(step_ft=1.3, at=20, n=160, fps=60.0):
    """SYNTHETIC: lead (left) toe steps sideways `step_ft` (x) as a sigmoid centred on frame `at`, then
    comes back toward the plate after a leg lift at frame 100 (z up)."""
    t = np.arange(n)
    j = np.zeros((n, 24, 3))
    j[:, 10, 0] = step_ft / (1 + np.exp(-(t - at) / 3.0))
    j[:, 10, 0] -= 3.0 / (1 + np.exp(-(t - 110) / 4.0))
    j[:, 7] = j[:, 10] + [0, 0.3, 0.2]
    return j


def test_rocker_step_found_at_half_the_step():
    from pitchsim.pose.delivery import rocker_step
    assert abs(rocker_step(_rocker_clip(), "R", leg_lift=100, fps=60.0) - 20) <= 1


def test_rocker_step_none_without_a_step():
    from pitchsim.pose.delivery import rocker_step
    assert rocker_step(_rocker_clip(step_ft=0.1), "R", leg_lift=100, fps=60.0) is None


def test_average_weights_fade_a_clip_out():
    """SYNTHETIC: clip b has weight 0 before 0.3 s pre-release, so early frames are clip a alone."""
    a, b = _clip(60, 120, 60, 0.05), _clip(60, 120, 60, 0.05)
    a[:, 0, 2], b[:, 0, 2] = 0.0, 1.0
    m = dict(leg_lift=20, foot_strike=40, release=60, fps=60.0)
    wb = lambda tau: (tau > -0.3).astype(float)  # noqa: E731
    j, ri = average_delivery([a, b], [m, m], pre_s=0.9, weights=[None, wb])
    assert np.allclose(j[: ri - 30, 0, 2], 0.0, atol=1e-6)
    assert np.allclose(j[ri - 10:, 0, 2], 0.5, atol=1e-6)


def _walk(n=120, fps=60.0):
    """SYNTHETIC legs: hips at +-0.1 m, each knee/ankle/toe swinging fore-aft out of phase (no real data)."""
    t = np.arange(n) / fps
    j = np.zeros((n, 24, 3))
    for side, (hip, knee, ank, toe), ph in (("L", (1, 4, 7, 10), 0.0), ("R", (2, 5, 8, 11), np.pi)):
        x = 0.1 if side == "L" else -0.1
        sw = 0.3 * np.sin(2 * np.pi * 1.0 * t + ph)
        j[:, hip] = [x, 0, 0.9]
        for k, z, s in ((knee, 0.5, 0.5), (ank, 0.1, 1.0), (toe, 0.0, 1.0)):
            j[:, k, 0] = x
            j[:, k, 1] = s * sw + (0.15 if k == toe else 0.0)
            j[:, k, 2] = z
    return j


def test_fix_leg_swaps_undoes_a_left_right_leg_swap():
    """SYNTHETIC: the monocular lift sometimes labels the left leg as the right (and vice versa) for a
    stretch of frames, then snaps back -- a one-frame 0.5-0.9 m jump of every leg joint.  Those swaps sat
    right before the leg lift in 5 real clips and made the average hitch there."""
    from pitchsim.pose.delivery import fix_leg_swaps
    j = _walk()
    bad = j.copy()
    bad[:40, [1, 4, 7, 10, 2, 5, 8, 11]] = j[:40, [2, 5, 8, 11, 1, 4, 7, 10]]
    fixed, n = fix_leg_swaps(bad, anchor_frame=100)
    assert n == 40
    assert np.allclose(fixed, j)


def test_fix_leg_swaps_leaves_a_clean_clip_alone():
    from pitchsim.pose.delivery import fix_leg_swaps
    j = _walk()
    fixed, n = fix_leg_swaps(j, anchor_frame=100)
    assert n == 0 and np.allclose(fixed, j)


def test_leg_swap_would_make_the_average_hitch():
    """SYNTHETIC regression: averaging a clip whose legs are swapped until some frame puts an acceleration
    spike in the average there (> 3x the clean average's); repaired first, it does not."""
    from pitchsim.pose.delivery import fix_leg_swaps
    clean = _walk(180)
    bad = clean.copy()
    bad[:70, [1, 4, 7, 10, 2, 5, 8, 11]] = clean[:70, [2, 5, 8, 11, 1, 4, 7, 10]]
    m = dict(leg_lift=60, foot_strike=120, release=150, fps=60.0)
    acc = lambda a: np.linalg.norm(np.diff(a, n=2, axis=0), axis=2).max()  # noqa: E731
    ref, _ = average_delivery([clean, clean], [m, m], pre_s=2.0, post_s=0.4)
    hitch, _ = average_delivery([clean, bad], [m, m], pre_s=2.0, post_s=0.4)
    rep, _ = average_delivery([clean, fix_leg_swaps(bad, 150)[0]], [m, m], pre_s=2.0, post_s=0.4)
    assert acc(hitch) > 3 * acc(ref)
    assert acc(rep) < 1.5 * acc(ref)


def test_lift_snap_finds_the_last_whole_leg_jump_before_release():
    """SYNTHETIC: a lift frozen in a wrong pose until frame 50, then snapping to the right one."""
    from pitchsim.pose.delivery import lift_snap
    j = _walk(180)
    j[:50, [4, 7, 10]] += [0, 0, -0.6]   # lead knee/ankle/toe 0.6 m off until the snap
    assert lift_snap(j, release=150, fps=60.0) == 50
    assert lift_snap(_walk(180), release=150, fps=60.0) is None


def test_valid_from_keeps_a_snap_out_of_the_average():
    """SYNTHETIC regression for the leg-lift hitch: one clip is garbage (0.6 m off) until its lift snaps at
    frame 60.  Averaged in, the snap is an acceleration spike far above the clean clips' own; with
    valid_from it fades in after the snap and the average stays as smooth as the clean one (< 1.5x)."""
    clean = _walk(180)
    bad = clean.copy()
    bad[:60, [4, 7, 10]] += [0, 0, -0.6]
    m = dict(leg_lift=60, foot_strike=120, release=150, fps=60.0)
    acc = lambda a: np.linalg.norm(np.diff(a, n=2, axis=0), axis=2).max()  # noqa: E731
    ref, _ = average_delivery([clean] * 3, [m] * 3, pre_s=2.0, post_s=0.4)
    hitch, _ = average_delivery([clean, clean, bad], [m] * 3, pre_s=2.0, post_s=0.4)
    ok, _ = average_delivery([clean, clean, bad], [m] * 3, pre_s=2.0, post_s=0.4, valid_from=[None, None, 60])
    assert acc(hitch) > 3 * acc(ref)
    assert acc(ok) < 1.5 * acc(ref)


def test_ground_root_does_not_copy_toe_jitter_into_the_body():
    """SYNTHETIC: pelvis travels smoothly (a parabola) over a planted pivot toe; the lift adds +-5 m root
    depth noise to the whole body and 1 cm of jitter to the toe joint alone.  Grounding must not hand that
    toe jitter to every joint (it showed as a whole-body kink 0.27 s before release in the average)."""
    rng = np.random.default_rng(1)
    n = 120
    t = np.arange(n) / 60.0
    j = np.zeros((n, 24, 3))
    j[:, 0] = np.stack([0.5 * t ** 2, np.zeros(n), np.full(n, 0.95)], 1)   # smooth pelvis path (m)
    for k in range(1, 24):
        j[:, k] = j[:, 0] + [0.01 * k, 0.0, -0.02 * k]                          # rigid body
    j[:, 11] = [0.0, 0.0, 0.0]                                                   # pivot toe planted
    noisy = j.copy()
    noisy[:, 11] += rng.normal(0, 0.01, (n, 3))                                  # toe jitter only
    noisy += np.stack([np.zeros(n), np.zeros(n), rng.normal(0, 5.0, n)], 1)[:, None, :]  # root depth noise
    g = ground_root(noisy, "R")
    err = (g[:, 0] - g[0, 0]) - (j[:, 0] - j[0, 0])
    assert np.sqrt(np.mean(np.diff(err, n=2, axis=0) ** 2)) < 0.25 * 0.01 * np.sqrt(6)


def test_ground_root_bridges_the_foot_hand_over_smoothly():
    """SYNTHETIC: the pelvis travels steadily toward the plate; in the lift the pivot toe slides 0.3 m
    forward relative to the pelvis over the 15 frames before foot strike (the real lifts do this as the
    back heel comes up), so pinning the pivot toe drags the body BACK and the switch to the lead foot at
    foot strike snaps it forward -- a whole-body kink ~0.1-0.3 s before release.  The root path across the
    hand-over must stay smooth (max second difference < 8 mm/frame^2)."""
    n, fs = 120, 80
    t = np.arange(n) / 60.0
    j = np.zeros((n, 24, 3))
    j[:, 0] = np.stack([np.zeros(n), -3.0 * t, np.full(n, 0.95)], 1)
    for k in range(1, 24):
        j[:, k] = j[:, 0] + [0.01 * k, 0.0, -0.02 * k]
    j[:, 11] = [0.0, 0.0, 0.0]                        # pivot toe on the rubber
    j[:, 10] = j[:, 0] + [0.2, -0.8, -0.95]           # lead toe swings with the body ...
    j[fs:, 10] = j[fs, 10]                            # ... and is planted from foot strike
    s = np.clip((np.arange(n) - (fs - 15)) / 15.0, 0, 1)
    j[:, 11, 1] -= 0.3 * s * s * (3 - 2 * s)          # lift artifact: pivot toe slides 0.3 m toward the plate
    g = ground_root(j, "R", fs)
    assert np.linalg.norm(np.diff(g[:, 0], n=2, axis=0), axis=1).max() < 0.008
