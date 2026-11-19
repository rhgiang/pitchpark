"""arm_refine tests.  ALL INPUTS ARE SYNTHETIC: a made-up body, a made-up horizontal-arm sweep and orthographic
cameras, not any pitcher's data."""
import numpy as np

from pitchsim.pose.anchor import _ELBOW, _HAND, _SHOULDER, _WRIST
from pitchsim.pose.arm_refine import Track, refine_arm
from pitchsim.pose.compare2d import COCO_TO_SMPL, SMPL_IDX, project

T, RI, FS = 80, 60, 51
LU, LF, LH = 1.0, 0.9, 0.3
L = "L"
SH, EL, WR, HA = _SHOULDER[L], _ELBOW[L], _WRIST[L], _HAND[L]


def _unit(a):
    return a / np.linalg.norm(a, axis=-1, keepdims=True)


def _true_arm(rng):
    """Body (T, 24, 3) with a throwing arm sweeping through the camera's viewing axis (y): phi 150 -> 30 deg
    in the x-y plane, depth sign constant (y > 0).  Returns (body, u_true, v_true)."""
    body = rng.normal(0, 1.0, (24, 3))[None].repeat(T, 0) + rng.normal(0, 0.02, (T, 24, 3)).cumsum(0) * 0.1
    body[:, 16], body[:, 17] = [-0.7, 0, 5], [0.7, 0, 5]     # shoulders
    body[:, 1], body[:, 2] = [-0.4, 0, 3], [0.4, 0, 3]       # hips
    t = np.arange(T)
    phi = np.radians(np.interp(t, [0, 40, 64, T - 1], [150, 150, 30, 30]))
    u = _unit(np.stack([np.cos(phi), np.sin(phi), np.full(T, -0.15)], -1))
    v = _unit(np.stack([np.cos(phi - 0.6), np.sin(phi - 0.6), np.full(T, -0.3)], -1))
    S = body[:, SH]
    body[:, EL] = S + LU * u
    body[:, WR] = body[:, EL] + LF * v
    body[:, HA] = body[:, WR] + LH * v
    return body, u, v


def _cameras():
    """Three orthographic cameras ~60 px/ft looking along y, each yawed a few degrees (different games)."""
    out = []
    for yaw in (0.0, 0.1, -0.12):
        c, s = np.cos(yaw), np.sin(yaw)
        out.append(60.0 * np.array([[c, s, 0.0], [0.0, 0.0, -1.0]]))
    return out


def _tracks(body, rng, noise_px=1.0):
    tracks = []
    for A in _cameras():
        b = np.cumsum(rng.normal(0, 1.0, (T, 2)), axis=0) + [640, 360]
        xy = project(A, b, body[:, SMPL_IDX]) + rng.normal(0, noise_px, (T, 12, 2))
        tracks.append(Track(xy, np.ones((T, 12), bool), RI, FS, 60.0))
    return tracks


def _collapse(body):
    """The lift's failure: the arm loses its depth (y) mid-swing, so the bones shrink and their direction jumps."""
    lift = body.copy()
    t = np.arange(T)
    g = 1 - 0.85 * np.exp(-((t - 52) / 2.0) ** 2)    # depth kept: 15% at t = 52, 100% away from it
    for k in (EL, WR, HA):
        r = body[:, k] - body[:, SH]
        lift[:, k] = body[:, SH] + r * np.stack([np.ones(T), g, np.ones(T)], -1)
    return lift


def _dirs(j):
    return _unit(j[:, EL] - j[:, SH]), _unit(j[:, WR] - j[:, EL])


def _turn_deg(a, b):
    return np.degrees(np.arccos(np.clip(np.sum(a * b, -1), -1, 1)))


def test_collapsed_depth_is_restored_with_fixed_lengths_and_smooth_path():
    rng = np.random.default_rng(0)
    body, u_true, v_true = _true_arm(rng)
    lift = _collapse(body)
    tracks = _tracks(body, rng)
    info = {}
    out = refine_arm(lift, RI, FS, L, tracks, info=info)
    u, v = _dirs(out)
    win = slice(FS - 6, RI + 7)       # the full-strength part of the window
    lu = np.linalg.norm(out[:, EL] - out[:, SH], axis=1)
    lf = np.linalg.norm(out[:, WR] - out[:, EL], axis=1)
    # the lift's upper arm shrank to ~0.15 of full length: the refinement gives the bone length back
    assert np.linalg.norm(lift[:, EL] - lift[:, SH], axis=1)[FS:RI + 1].min() < 0.5 * LU
    # the bones are back at their standing length (the window is the full-strength part)
    np.testing.assert_allclose(lu[win], LU, atol=1e-6)
    np.testing.assert_allclose(lf[win], LF, atol=1e-6)
    # the direction is right again, sign included (continuity from the well-conditioned frames)
    assert _turn_deg(u, u_true)[win].max() < 12
    assert _turn_deg(v, v_true)[win].max() < 15
    # and the path is smoother than the lift's whip
    u_l, _ = _dirs(lift)
    assert _turn_deg(u[1:], u[:-1])[win].max() < 0.5 * _turn_deg(u_l[1:], u_l[:-1])[win].max()
    # the hand stays on the release point correct_to_mean set, and nothing moved outside the window
    np.testing.assert_allclose(out[RI, HA], lift[RI, HA], atol=1e-3)
    np.testing.assert_array_equal(out[:FS - 12], lift[:FS - 12])
    assert info["data_rms_after"] < info["data_rms_before"]


def test_well_conditioned_arm_is_left_almost_unchanged():
    rng = np.random.default_rng(1)
    body, u_true, v_true = _true_arm(rng)
    tracks = _tracks(body, rng)
    out = refine_arm(body, RI, FS, L, tracks)
    u, v = _dirs(out)
    assert _turn_deg(u, u_true).max() < 4
    assert _turn_deg(v, v_true).max() < 5
    assert np.abs(out - body).max() < 0.15


def test_no_tracks_returns_input():
    rng = np.random.default_rng(2)
    body, _, _ = _true_arm(rng)
    np.testing.assert_array_equal(refine_arm(body, RI, FS, L, []), body)


def test_more_data_weight_pulls_the_arm_toward_the_2d():
    """SYNTHETIC: the lift's arm is turned 20 deg in the image plane from the true one; the tracks see the true
    arm.  The lift prior holds the arm back; a larger data_w lets the 2D win, so the arm ends up nearer the truth."""
    rng = np.random.default_rng(3)
    body, u_true, v_true = _true_arm(rng)
    tracks = _tracks(body, rng)[:1]      # one clip, a strong prior: the data has to be weighted up to win
    lift = body.copy()
    a = np.radians(20)
    R = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])   # about y: the x-z plane
    for k in (EL, WR, HA):
        lift[:, k] = body[:, SH] + (body[:, k] - body[:, SH]) @ R.T
    err = lambda out: _turn_deg(_dirs(out)[0], u_true)[FS - 6:RI + 7].mean()   # noqa: E731
    assert err(refine_arm(lift, RI, FS, L, tracks, data_w=4.0, pin=0.0, hand_w=0.0)) <         err(refine_arm(lift, RI, FS, L, tracks, data_w=1.0, prior=2.0, pin=0.0, hand_w=0.0))


def test_release_reach_ratio_flags_a_foreshortened_arm():
    """SYNTHETIC: a straight arm at release has ratio ~1; the same arm with the forearm folded back has less."""
    from pitchsim.pose.arm_refine import release_reach_ratio
    rng = np.random.default_rng(4)
    body, _, _ = _true_arm(rng)
    straight = body.copy()
    straight[:, EL] = straight[:, SH] + [LU, 0, 0]
    straight[:, WR] = straight[:, EL] + [LF, 0, 0]
    straight[:, HA] = straight[:, WR] + [LH, 0, 0]
    assert abs(release_reach_ratio(straight, RI, L) - 1.0) < 1e-6
    folded = straight.copy()
    folded[RI, WR] = folded[RI, EL] + [-LF * 0.5, 0, LF * 0.8]       # forearm turned back on itself at release
    folded[RI, HA] = folded[RI, WR] + [-LH * 0.5, 0, LH * 0.8]
    assert release_reach_ratio(folded, RI, L) < 0.6
