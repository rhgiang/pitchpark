"""king_smooth on SYNTHETIC data (a made-up 24-joint skeleton with noise), not a real pitcher."""
import numpy as np

from pitchsim.pose.anchor import SMPL_PARENT, segment_lengths
from pitchsim.pose.king_smooth import smooth_king

N, FPS, LIFT, REL = 240, 60.0, 100, 200


def _skeleton(seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(N) / FPS
    base = rng.normal(size=(24, 3))
    j = np.zeros((N, 24, 3))
    j[:, 0] = np.stack([0.3 * t, 0 * t, 3 + 0.0 * t], 1)
    for k in range(1, 24):  # fixed bone offsets plus a slow wave, then noise on the angles
        off = base[k] / np.linalg.norm(base[k]) * 0.8
        wave = 0.15 * np.stack([np.sin(2 * t + k), np.cos(3 * t + k), np.sin(t)], 1)
        d = off + wave + rng.normal(scale=0.03, size=(N, 3))  # noisy direction ...
        d *= 0.8 / np.linalg.norm(d, axis=1)[:, None]  # ... but constant bone length, as in a real skeleton
        j[:, k] = j[:, SMPL_PARENT[k]] + d
    return j


def test_king_smooth_invariants():
    j = _skeleton()
    o = smooth_king(j, FPS, LIFT, REL, "R")
    assert np.abs(o[REL] - j[REL]).max() == 0          # release pose exact
    assert np.abs(o[REL - 1:] - j[REL - 1:]).max() == 0  # hold frame and after untouched
    feet = [7, 8, 10, 11]
    assert np.abs(o[REL - 24:, feet] - j[REL - 24:, feet]).max() == 0  # legs not touched in the last 0.4 s
    assert np.abs(segment_lengths(o) - segment_lengths(j)).max() < 1e-9
    assert np.abs(o[LIFT + 20:REL - 30] - j[LIFT + 20:REL - 30]).max() == 0  # between the two passes: bit-identical


def test_king_smooth_reduces_jerk():
    j = _skeleton()
    o = smooth_king(j, FPS, LIFT, REL, "R")
    jerk = lambda x: np.abs(np.diff(x, 3, axis=0)).max()
    assert jerk(o[:LIFT - 30, 3:]) < jerk(j[:LIFT - 30, 3:])
    assert jerk(o[REL - 24:REL - 3, 3:]) < jerk(j[REL - 24:REL - 3, 3:])
