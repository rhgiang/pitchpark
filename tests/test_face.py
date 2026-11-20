"""M7 Task 4: aiming, re-aim reference. Synthetic knots / physics only (no real data)."""
import numpy as np
import pytest

from pitchsim.face import aim_velocity, plate_crossing, reaim_knots
from pitchsim.physics import PhysicsParams, simulate

# Synthetic two-knot flight (not a real pitch): release at y=54, plate end at y=1.5.
SIM = {"s": [[0, 54, 6, 0, -140, -5], [0.1, 1.5, 2.3, 1, -128, -15]], "T": 0.42}


@pytest.fixture
def sim():
    return {"s": [list(r) for r in SIM["s"]] , "T": SIM["T"]}


def test_aim_hits_target():
    p = PhysicsParams(cd=0.33, spin_efficiency=0.9, rho=0.0740)
    v = aim_velocity(np.array([-1.8, 54.0, 5.9]), 140.0, 2300.0, 210.0, p, np.zeros(2), (0.3, 2.2))
    tr = simulate(np.array([-1.8, 54.0, 5.9]), v, 2300.0, 210.0, p)
    x, z = plate_crossing(tr)
    assert abs(x - 0.3) * 12 < 0.01 and abs(z - 2.2) * 12 < 0.01
    assert np.isclose(np.linalg.norm(v), 140.0)


def test_reaim_zero_offset_is_identity(sim):
    assert reaim_knots(sim, 0.0, 0.0) == sim


def test_reaim_moves_plate_end_exactly_and_release_not_at_all(sim):
    out = reaim_knots(sim, 0.5, -0.25)
    assert out["s"][0][:3] == sim["s"][0][:3]
    assert np.allclose(np.array(out["s"][-1][:3]) - sim["s"][-1][:3], [0.5, 0, -0.25])
    assert out["T"] == sim["T"]


def test_far_target_reaims_to_exact_plate_point(sim):
    out = reaim_knots(sim, 3.0, 0.0)
    s = np.array(out["s"])
    assert np.isfinite(s).all()
    assert np.isclose(s[-1][0], sim["s"][-1][0] + 3.0)
    assert out["T"] == sim["T"]
    assert np.allclose(s[:, 1], np.array(sim["s"])[:, 1])   # y untouched
