import numpy as np
import pandas as pd

from pitchsim.generative import (FEATURES, add_release_angles, circular_mean_deg,
                                 fit_pitch_type, sample)
from pitchsim.synthetic import make


def test_sample_matches_fitted_gaussian():
    """Synthetic: draws from a known 8-D Gaussian; refit+sample recovers mean and cov."""
    rng = np.random.default_rng(0)
    mean = np.array([-1.9, 5.9, 6.3, 95, 2400, 210, -1.5, 1.0])
    cov = np.diag([0.01, 0.01, 0.02, 0.6, 3600, 36, 0.3, 0.3])
    train = pd.DataFrame(rng.multivariate_normal(mean, cov, 5000), columns=FEATURES)
    out = sample(fit_pitch_type(train, mixture=False), 20000, seed=1)
    assert np.allclose(out[FEATURES].mean(), mean, rtol=0.01, atol=0.05)
    assert np.allclose(out[FEATURES].std(), np.sqrt(np.diag(cov)), rtol=0.05)


def test_spin_axis_wraps_across_zero():
    """Synthetic: axis centered on 355 deg, spread 10 deg."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(0, 1, (3000, 8)), columns=FEATURES)
    df["spin_axis"] = (355 + rng.normal(0, 10, 3000)) % 360
    m = fit_pitch_type(df, mixture=False)
    assert abs(((m.axis_center_deg - 355) + 180) % 360 - 180) < 1
    out = sample(m, 5000, seed=1)["spin_axis"]
    assert out.between(0, 360, inclusive="left").all()
    assert abs(((circular_mean_deg(out.to_numpy()) - 355) + 180) % 360 - 180) < 1


def test_mixture_finds_two_clusters_and_respects_minimum():
    """Synthetic: two well-separated velocity clusters."""
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, (400, 8)); b = rng.normal(0, 1, (400, 8)); b[:, 3] += 8
    df = pd.DataFrame(np.vstack([a, b]), columns=FEATURES); df["spin_axis"] += 200
    assert fit_pitch_type(df, mixture=True).gmm.n_components == 2
    assert fit_pitch_type(df.iloc[:60], mixture=True).gmm.n_components == 1


def test_release_angles_from_synthetic():
    """Synthetic pitcher: release angles are small, downward-ish."""
    df = add_release_angles(make(n=200, seed=0))
    assert df["release_angle_v_deg"].between(-10, 5).all()
