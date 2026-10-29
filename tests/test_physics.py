"""Sanity checks against well-known pitch physics numbers.  Run: pytest -q"""

import numpy as np

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
