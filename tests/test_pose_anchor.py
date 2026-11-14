"""Task 4 tests, anchor.py.  ALL INPUTS ARE SYNTHETIC (a made-up standing figure), not real pitcher data."""
import numpy as np
from scipy.spatial.transform import Rotation

from pitchsim.pose.anchor import anchor, correct_to_mean, release_values

RUBBER_Y, MOUND_Z = 60.5, 10 / 12


def _figure(n=30):
    """1.8 m tall figure facing +z with up = +y and left = +x (the SMPL rest convention), static."""
    j = np.zeros((n, 24, 3))
    j[:, 1] = [0.1, 0.9, 0]; j[:, 2] = [-0.1, 0.9, 0]; j[:, 0] = [0, 0.95, 0]
    j[:, 7] = [0.1, 0.08, 0]; j[:, 8] = [-0.1, 0.08, 0]
    j[:, 10] = [0.1, 0.0, 0.15]; j[:, 11] = [-0.1, 0.0, 0.15]
    j[:, 15] = [0, 1.8 / 1.07, 0]
    j[:, 16] = [0.2, 1.45, 0]; j[:, 17] = [-0.2, 1.45, 0]
    return j


def _rotated(j):
    r = Rotation.random(random_state=0)
    return r.apply(j.reshape(-1, 3)).reshape(j.shape) + [3.0, -2.0, 5.0], r.apply([0, 1, 0])


def test_anchor_height_and_rubber():
    j, up = _rotated(_figure())
    out = anchor(j, up, 6.25, "R", release_index=15)
    head, feet = out[:10, 15], out[:10][:, [10, 11]]
    h = 1.07 * np.median(np.linalg.norm(head - feet[np.arange(10), np.argmin(feet[..., 2], 1)], axis=1))
    assert abs(h - 6.25) < 0.05
    assert np.allclose(out[0, 11], [0, RUBBER_Y, MOUND_Z], atol=0.01)
    fwd = np.cross(out[15, 1] - out[15, 2], [0, 0, 1.0])
    fwd /= np.linalg.norm(fwd)
    assert fwd[1] < -0.99


def test_feet_do_not_slide():
    f = _figure(60)
    f[:, [8, 11], 0] += np.linspace(0, 0.01, 60)[:, None]  # right foot drifts 1 cm in x (synthetic)
    j, up = _rotated(f)
    out = anchor(j, up, 6.25, "R", release_index=30, fps=60.0)
    for k in (8, 11):
        assert np.ptp(out[:, k, :2], axis=0).max() < 0.01


def _seq(release=30, n=40):
    j = np.zeros((n, 24, 3))
    j[:, 11] = [0, RUBBER_Y, MOUND_Z]
    s = np.clip(np.arange(n) / release, 0, 1)
    sh = np.stack([np.full(n, -0.8), 60.0 - 3.0 * s, np.full(n, 5.2)], 1)
    back, front = np.array([0, 1.8, 0.8]), 2.2 * np.array([0.1, -0.8, 0.59])
    hand = sh + (1 - s)[:, None] * back + s[:, None] * front
    j[:, 17] = sh; j[:, 23] = hand
    j[:, 19] = sh + 0.4 * (hand - sh); j[:, 21] = sh + 0.8 * (hand - sh)
    return j


def test_release_values():
    j = np.zeros((3, 24, 3)); j[:, 17] = [0, 60, 5]
    j[:, 23] = [0, 60, 7.2]
    assert abs(release_values(j, 1, "R")["arm_angle_deg"] - 90) < 1e-6
    j[:, 23] = [2, 60, 5]
    assert abs(release_values(j, 1, "R")["arm_angle_deg"]) < 1e-6
    j[:, 17] = [-1, 58, 6]; j[:, 23] = [-2, 56, 7]  # forward reach (dy != 0) is ignored: catcher's view
    assert abs(release_values(j, 1, "R")["arm_angle_deg"] - 45) < 1e-6
    j[:, 17] = [0, 60, 5]
    j[:, 23] = [2, 54.0, 5]
    assert abs(release_values(j, 1, "R")["extension_ft"] - 6.5) < 1e-9


def test_correct_to_mean():
    j = _seq()
    rv = release_values(j, 30, "R")
    target = dict(rv, release_x_ft=rv["release_x_ft"] + 0.5, release_z_ft=rv["release_z_ft"] - 0.5,
                  extension_ft=rv["extension_ft"] + 0.5, arm_angle_deg=rv["arm_angle_deg"] + 4.0)
    out = correct_to_mean(j, 30, 20, "R", target)
    got = release_values(out, 30, "R")
    for k in ("release_x_ft", "release_z_ft", "extension_ft"):
        assert abs(got[k] - target[k]) < 0.05, k
    assert abs(got["arm_angle_deg"] - target["arm_angle_deg"]) < 1.0
    assert np.all(np.isfinite(out))
