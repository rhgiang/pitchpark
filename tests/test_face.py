"""M7 Task 4: aiming, re-aim reference. Synthetic knots / physics only (no real data)."""
import numpy as np
import pandas as pd
import pytest

from pitchsim.face import aim_velocity, make_bank, plate_crossing, reaim_error_in, reaim_knots
from pitchsim.seam import TypeFit, fit_fixed
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


def test_reaim_velocity_shift_is_derivative_of_position_shift(sim):
    out = reaim_knots(sim, 0.5, -0.25)
    y0, y1 = sim["s"][0][1], sim["s"][-1][1]
    for a, b in zip(sim["s"], out["s"]):
        k = -a[4] / (y0 - y1)
        assert np.allclose(np.array(b[3:]) - a[3:], [0.5 * k, 0, -0.25 * k])


def test_reaim_degenerate_knots_raise():
    with pytest.raises(ValueError):
        reaim_knots({"s": [[0, 5, 1, 0, -1, 0], [0, 5, 1, 0, -1, 0]], "T": 0.1}, 1.0, 0.0)


def test_aim_nonconverged_solution_raises(monkeypatch):
    """A solver that stops 1 deg off the target must raise, not return a bad velocity."""
    from types import SimpleNamespace
    import pitchsim.face as face
    monkeypatch.setattr(face, "least_squares",
                        lambda f, x0, **kw: SimpleNamespace(x=np.asarray(x0) + 1.0))
    p = PhysicsParams(cd=0.33, spin_efficiency=0.9, rho=0.0740)
    with pytest.raises(ValueError, match="did not converge"):
        aim_velocity(np.array([-1.8, 54.0, 5.9]), 140.0, 2300.0, 210.0, p, np.zeros(2), (0.3, 2.2))


def _synthetic_fits_and_train():
    """Synthetic (not real) FF-like frame and a fixed-push fit; for plumbing tests only."""
    rng = np.random.default_rng(0)
    n = 40
    train = pd.DataFrame({
        "pitch_type": "FF", "release_pos_x": rng.normal(-1.8, .1, n),
        "release_pos_z": rng.normal(5.9, .1, n), "release_extension": rng.normal(6.5, .2, n),
        "release_speed": rng.normal(95, 1, n), "release_spin_rate": rng.normal(2300, 50, n),
        "spin_axis": rng.normal(210, 5, n), "release_angle_v_deg": rng.normal(-1, .3, n),
        "release_angle_h_deg": rng.normal(1, .3, n)})
    fits = {"FF": TypeFit(0.33, 0.9, fit_fixed(np.zeros((n, 2))), fit_fixed(np.zeros((n, 2))))}
    return train, fits


def test_make_bank_entries_hit_aim_and_have_keys():
    train, fits = _synthetic_fits_and_train()
    bank = make_bank(train, 0.074, fits, {"FF": (0.1, 2.5)}, n_per_type=3, seed=0)
    assert len(bank["FF"]) == 3
    for e in bank["FF"]:
        assert {"sim", "plate", "aim", "release", "speed_mph", "spin_rpm", "spin_axis_deg",
                "extension_ft", "break_in"} <= set(e)
        assert np.hypot(e["plate"][0] - e["aim"][0], e["plate"][1] - e["aim"][1]) * 12 < 0.01


def test_reaim_error_small_for_1_to_2_ft_offsets():
    train, fits = _synthetic_fits_and_train()
    bank = make_bank(train, 0.074, fits, {"FF": (0.0, 2.5)}, n_per_type=2, seed=1)
    params = {"FF": PhysicsParams(cd=0.33, spin_efficiency=0.9, rho=0.074)}
    tg = {"FF": np.array([[1.5, 1.5], [-2.0, 3.0]])}      # offsets (1.5, -1.0) and (-2.0, 0.5)
    err = reaim_error_in(bank, tg, params)
    assert (err["path_in"] < 0.5).all() and (err["plate_in"] < 0.01).all()
