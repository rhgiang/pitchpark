"""Sanity checks against well-known pitch physics numbers.  Run: pytest -q"""

import numpy as np
import pytest

from pitchsim.physics import PhysicsParams, air_density, simulate, simulate_batch

MPH = 1 / 0.681818
RELEASE = np.array([0.0, 54.0, 6.0])


def fastball(spin=2400, axis=180, eff=1.0, cd=0.33):
    v = 95 * MPH * np.array([0.0, -1.0, -0.03]) / np.linalg.norm([0, -1, -0.03])
    return simulate(RELEASE, v, spin, axis, PhysicsParams(cd=cd, spin_efficiency=eff))


def test_flight_time_about_four_tenths():
    t = fastball()["t"][-1]
    assert 0.38 < t < 0.45


def test_drag_costs_several_mph():
    tr = fastball()
    loss = 1 - np.linalg.norm(tr["vel"][-1]) / np.linalg.norm(tr["vel"][0])
    assert 0.06 < loss < 0.11          # ~8% speed loss release -> plate


def test_backspin_rise_is_realistic():
    """Induced vertical break of a 2400 rpm, full-backspin fastball is ~16-20 in."""
    with_spin = fastball(eff=1.0)["pos"][-1, 2]
    no_spin = fastball(eff=0.0)["pos"][-1, 2]
    ivb = (with_spin - no_spin) * 12
    assert 13 < ivb < 22


def test_spin_axis_direction():
    """90 deg moves toward +x, 270 toward -x, 0 (topspin) moves down."""
    base = fastball(eff=0.0)["pos"][-1]
    assert fastball(axis=90)["pos"][-1, 0] > base[0] + 0.5
    assert fastball(axis=270)["pos"][-1, 0] < base[0] - 0.5
    assert fastball(axis=0)["pos"][-1, 2] < base[2] - 0.5


def test_thin_air_reduces_break():
    hi = PhysicsParams(rho=air_density(70, 5200))
    v = 80 * MPH * np.array([0.0, -1.0, 0.0])
    denver = simulate(RELEASE, v, 2600, 0, hi)["pos"][-1, 2]
    sea = simulate(RELEASE, v, 2600, 0, PhysicsParams())["pos"][-1, 2]
    assert denver > sea                 # curveball drops less at Coors


def test_batch_matches_single():
    v = 90 * MPH * np.array([0.05, -1.0, -0.02])
    single = simulate(RELEASE, v, 2200, 220, PhysicsParams(cd=0.33, spin_efficiency=0.9))
    pos, _, t = simulate_batch(RELEASE[None], v[None], [2200], [220], 0.33, 0.9)
    assert np.allclose(pos[0], single["pos"][-1], atol=1e-6)
    assert abs(t[0] - single["t"][-1]) < 1e-6


def test_batch_per_pitch_density():
    v = 80 * MPH * np.array([0.0, -1.0, 0.0])
    sea, coors = air_density(70, 0), air_density(70, 5200)
    pos, _, _ = simulate_batch(np.stack([RELEASE, RELEASE]), np.stack([v, v]),
                               [2600, 2600], [0, 0], 0.33, 0.9, rho=np.array([sea, coors]))
    for i, rho in enumerate([sea, coors]):
        one = simulate(RELEASE, v, 2600, 0, PhysicsParams(cd=0.33, spin_efficiency=0.9, rho=rho))
        assert np.allclose(pos[i], one["pos"][-1], atol=1e-6)


def _batch(extra_acc=None, n=3):
    v = 95 * MPH * np.array([0.0, -1.0, -0.03]) / np.linalg.norm([0, -1, -0.03])
    kw = {} if extra_acc is None else {"extra_acc": extra_acc}
    return simulate_batch(np.tile(RELEASE, (n, 1)), np.tile(v, (n, 1)),
                          np.full(n, 2400.0), np.full(n, 200.0), 0.33, 0.9, **kw)


def test_extra_acc_none_is_identical():
    base = _batch()
    for extra in (None, np.zeros((3, 2))):
        for a, b in zip(base, _batch(extra)):
            assert np.array_equal(a, b)


def test_extra_acc_shifts_by_half_a_t_squared():
    pos0, _, t = _batch(n=1)
    pos1, _, _ = _batch(np.array([[3.0, -2.0]]), n=1)
    t = t[0]
    # drag damps the push by ~k*Cd*v*t/3 ~ 3% (measured 3.0%), so 5% not 2%
    assert (pos1[0, 0] - pos0[0, 0]) == pytest.approx(0.5 * 3.0 * t ** 2, rel=0.05)
    assert (pos1[0, 2] - pos0[0, 2]) == pytest.approx(0.5 * -2.0 * t ** 2, rel=0.05)


def test_single_and_batch_agree_with_push():
    v = 95 * MPH * np.array([0.0, -1.0, -0.03]) / np.linalg.norm([0, -1, -0.03])
    push = np.array([3.0, -2.0])
    tr = simulate(RELEASE, v, 2400, 200, PhysicsParams(cd=0.33, spin_efficiency=0.9),
                  extra_acc=push)
    pos, _, _ = simulate_batch(RELEASE[None], v[None], [2400.0], [200.0], 0.33, 0.9,
                               extra_acc=push[None])
    assert np.allclose(tr["pos"][-1], pos[0], atol=1e-6)
