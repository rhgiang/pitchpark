"""Task 4 tests, anchor.py.  ALL INPUTS ARE SYNTHETIC (a made-up standing figure), not real pitcher data."""
import numpy as np
from scipy.spatial.transform import Rotation

from pitchsim.pose.anchor import anchor, correct_to_mean, mound_z_ft, release_values, segment_lengths

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


def _delivery(n=60, ll=15, fs=35, ri=45):
    """SYNTHETIC right-hander delivery in Statcast ft (made-up stick figure, not real data): stance, leg
    lift peaking at `ll`, a short 1.4 ft stride landing at `fs`, arm whip to release at `ri`.  Legs are
    roughly jointed (knee bulging forward); segment lengths vary a little per frame, as in a real average."""
    t = np.arange(n, dtype=float)
    ss = lambda a, b: np.clip((t - a) / (b - a), 0, 1) ** 2 * (3 - 2 * np.clip((t - a) / (b - a), 0, 1))
    j = np.zeros((n, 24, 3))
    pel = np.stack([np.zeros(n), 60.3 - 0.9 * ss(ll, ri), 3.4 - 0.2 * ss(ll, fs)], 1)
    j[:, 0] = pel
    j[:, 1] = pel + [0.3, 0, -0.15]; j[:, 2] = pel + [-0.3, 0, -0.15]
    for k, dz in ((3, 0.35), (6, 0.8), (9, 1.05), (12, 2.0), (15, 2.3)):
        j[:, k] = pel + [0, 0, dz]
    j[:, 13] = pel + [0.25, 0, 1.75]; j[:, 14] = pel + [-0.25, 0, 1.75]
    j[:, 16] = pel + [0.62, 0, 1.8]; j[:, 17] = pel + [-0.62, 0, 1.8]
    j[:, 18] = j[:, 16] + [0.3, -0.6, -0.6]; j[:, 20] = j[:, 18] + [0, -0.8, 0.1]; j[:, 22] = j[:, 20] + [0, -0.25, 0]
    s = np.clip(t / ri, 0, 1)[:, None]
    back, front = np.array([-0.5, 1.8, 0.8]), np.array([-1.2, -1.5, 1.2])
    j[:, 23] = j[:, 17] + (1 - s) * back + s * front
    j[:, 19] = j[:, 17] + 0.45 * (j[:, 23] - j[:, 17]) + [0, 0.3, -0.2]
    j[:, 21] = j[:, 17] + 0.88 * (j[:, 23] - j[:, 17])
    # pivot (right) toe pinned on the rubber until foot strike, then drags 0.5 ft toward the plate
    j[:, 11] = [0.0, RUBBER_Y, MOUND_Z]
    j[:, 11, 1] -= 0.5 * ss(fs, n - 1)
    # lead (left) toe: stance, lifted to 1.6 ft at ll, lands 1.4 ft in front of the pivot at fs
    lift = np.where(t < ll, ss(ll - 10, ll), 1 - ss(ll, fs))
    j[:, 10] = np.stack([0.9 - 0.3 * ss(ll, fs), 60.4 - 1.5 * ss(ll, fs), MOUND_Z + 1.6 * lift], 1)
    for toe, ank, hip, knee in ((10, 7, 1, 4), (11, 8, 2, 5)):
        j[:, ank] = j[:, toe] + [0, 0.45, 0.2]
        j[:, knee] = 0.5 * (j[:, hip] + j[:, ank]) + [0, -0.35, 0]
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


def _corrected():
    j = _delivery()
    rv = release_values(j, 45, "R")
    target = dict(rv, release_x_ft=rv["release_x_ft"] + 0.3, release_z_ft=rv["release_z_ft"] - 0.4,
                  extension_ft=rv["extension_ft"] + 2.0, arm_angle_deg=rv["arm_angle_deg"] + 5.0)
    return j, correct_to_mean(j, 45, 35, "R", target, leg_lift_index=15), target


def test_correct_to_mean():
    """SYNTHETIC: release values reach a target 2 ft further out, 0.4 ft lower, 0.3 ft over and 5 deg up."""
    _, out, target = _corrected()
    got = release_values(out, 45, "R")
    for k in ("release_x_ft", "release_z_ft", "extension_ft"):
        assert abs(got[k] - target[k]) < 0.05, k
    assert abs(got["arm_angle_deg"] - target["arm_angle_deg"]) < 1.0
    assert np.all(np.isfinite(out))


def test_correct_to_mean_keeps_segment_lengths():
    """SYNTHETIC: no bone changes length (1 mm), frame by frame -- the correction moves joints rigidly or
    re-solves legs with two-bone IK at each frame's own thigh and shin lengths."""
    j, out, _ = _corrected()
    assert np.abs(segment_lengths(out) - segment_lengths(j)).max() < 0.001 / 0.3048


def test_correct_to_mean_feet_planted():
    """SYNTHETIC: the pivot toe stays on the rubber until foot strike and the lead toe stays where it
    (now further out) landed from foot strike on, both exactly on the mound surface."""
    _, out, _ = _corrected()
    for toe, frames in ((11, slice(0, 36)), (10, slice(35, None))):
        p = out[frames, toe]
        assert np.ptp(p, axis=0).max() < 0.001, toe                       # no slide
        assert np.abs(p[:, 2] - mound_z_ft(p[:, 0], p[:, 1])).max() < 0.001, toe  # no sink / float
    assert out[40, 10, 1] < _delivery()[40, 10, 1] - 0.5  # the stride really got longer


def test_correct_to_mean_is_smooth():
    """SYNTHETIC: the correction (out - in) eases in; no joint jumps by more than 0.25 ft between frames
    because of it (a snap onto the release point would)."""
    j, out, _ = _corrected()
    assert np.abs(np.diff(out - j, axis=0)).max() < 0.25
