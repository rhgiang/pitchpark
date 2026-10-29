"""Tests for release_state. All data is SYNTHETIC (pitchsim.synthetic), not real Statcast."""
import numpy as np

from pitchsim.physics import simulate_batch
from pitchsim.statcast import initial_state, release_state
from pitchsim.synthetic import TRUE_CD, make


def test_release_state_position_is_release_point():
    """SYNTHETIC data: position equals release_pos_* by construction."""
    df = make(n=300, seed=0)
    pos, _ = release_state(df)
    assert np.allclose(pos, df[["release_pos_x", "release_pos_y", "release_pos_z"]].to_numpy())


def test_release_state_velocity_runs_fit_backwards():
    """SYNTHETIC data. Release speed > speed at y=50 (drag slows the ball), by under ~2 mph over ~5 ft."""
    df = make(n=300, seed=0)
    _, v_rel = release_state(df)
    _, v_50 = initial_state(df)
    dv = (np.linalg.norm(v_rel, axis=1) - np.linalg.norm(v_50, axis=1)) * 0.681818
    assert np.all(dv > 0) and np.all(dv < 2)


def test_release_start_recovers_synthetic_plate_location():
    """SYNTHETIC data. Simulating from release with the TRUE physics lands near plate_x/z.
    Residual is the constant-acceleration fit's own approximation error."""
    df = make(n=300, seed=0)
    pos0, vel0 = release_state(df)
    pos, _, _ = simulate_batch(pos0, vel0, df["release_spin_rate"], df["spin_axis"],
                               TRUE_CD, df["true_spin_eff"])
    err_in = np.hypot(pos[:, 0] - df["plate_x"], pos[:, 2] - df["plate_z"]) * 12
    assert np.median(err_in) < 1.0
