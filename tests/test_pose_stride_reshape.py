"""reshape_stride on SYNTHETIC data (a made-up lead leg with a fast extend-then-re-bend), not a real pitcher."""
import numpy as np

from pitchsim.pose.anchor import _ANKLE, _HIP, _KNEE, _TOE, segment_lengths
from pitchsim.pose.stride_reshape import knee_angle_deg, reshape_stride

N, A, B = 40, 10, 30


def _leg():
    j = np.zeros((N, 24, 3))
    t = np.arange(N)
    ang = np.radians(50 + 90 * np.exp(-((t - 18) / 3.0) ** 2))  # sharp extension peak at 18
    lead = "L"
    j[:, _HIP[lead]] = [0, 0, 3]
    j[:, _KNEE[lead]] = j[:, _HIP[lead]] + [0, 1, 0]
    # shank direction rotates in the y-z plane so the hip-knee-ankle angle is `ang`
    j[:, _ANKLE[lead], 1] = 1 - np.cos(ang)  # y
    j[:, _ANKLE[lead], 2] = 3 - np.sin(ang)  # z
    j[:, _TOE[lead]] = j[:, _ANKLE[lead]] + [0.2, 0, 0]
    return j


def test_reshape_invariants_synthetic():
    j = _leg()
    o = reshape_stride(j, A, B, "R", lam=30, fall_from=18, fall_weight=0.1)
    assert np.array_equal(o[:A], j[:A]) and np.array_equal(o[B:], j[B:])
    assert np.abs(segment_lengths(o) - segment_lengths(j)).max() < 1e-9
    a0, a1 = knee_angle_deg(j, "L"), knee_angle_deg(o, "L")
    assert np.abs(np.diff(a1, 2)).max() < np.abs(np.diff(a0, 2)).max()  # gentler
    assert np.allclose(a1[A:A + 3], a0[A:A + 3]) and np.allclose(a1[B - 2:B + 1], a0[B - 2:B + 1])
