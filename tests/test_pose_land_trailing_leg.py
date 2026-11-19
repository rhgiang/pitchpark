"""land_trailing_leg on SYNTHETIC data (a made-up standing skeleton whose back foot stays kicked up), not a real pitcher."""
import numpy as np

from pitchsim.pose.anchor import land_trailing_leg, mound_z_ft, segment_lengths

N, REL, START, LAND = 60, 20, 22, 48
HIP, KNEE, ANK, TOE = 2, 5, 8, 11


def _kicked():
    """Pelvis 3.2 ft up over a flat spot (y = 59); right leg 1.5 + 1.5 ft, foot 0.5 ft, swung up behind."""
    j = np.zeros((N, 24, 3))
    j[:] = [0.0, 59.0, 3.2]
    j[:, HIP] += [0.3, 0, 0]
    ang = np.radians(np.linspace(20, 100, N))            # the leg trails further and further back, never lands
    for k in range(N):
        d = np.array([0.0, np.sin(ang[k]), -np.cos(ang[k])])
        j[k, KNEE] = j[k, HIP] + 1.5 * d
        j[k, ANK] = j[k, KNEE] + 1.5 * d * [1, 1, 0.6] / np.linalg.norm(d * [1, 1, 0.6])
        j[k, TOE] = j[k, ANK] + [0.0, 0.5, 0.0]
    return j


def test_leg_lands_and_stays_planted():
    j = _kicked()
    o = land_trailing_leg(j, REL, START, LAND, "R")
    assert np.array_equal(o[:START + 1], j[:START + 1])          # nothing before the blend moves
    assert np.abs(o[LAND:, ANK] - o[LAND, ANK]).max() == 0       # no sliding once planted
    assert np.abs(o[LAND:, TOE] - o[LAND, TOE]).max() == 0
    ground = mound_z_ft(*o[LAND, ANK, :2])
    assert abs(o[LAND, ANK, 2] - ground - 0.305) < 1e-6 or o[LAND, ANK, 2] > ground  # on (or just above) the ground
    assert o[LAND:, TOE, 2].min() > ground - 0.05                # no ground penetration
    lens = segment_lengths(o)[:, [4, 7, 10]]                     # right thigh, shin, foot: constant after landing
    assert np.abs(lens[LAND:] - lens[LAND]).max() < 1e-9
    upper = [i for i in range(24) if i not in (KNEE, ANK, TOE)]
    assert np.array_equal(o[:, upper], j[:, upper])              # only the three leg joints move
    d = np.linalg.norm(o[:, ANK] - o[:, HIP], axis=1)
    assert d.max() < lens[:, 0].max() + lens[:, 1].max()         # never hyperextended
