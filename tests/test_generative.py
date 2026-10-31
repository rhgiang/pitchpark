import numpy as np
import pandas as pd

from pitchsim.generative import (FEATURES, GRADED, add_release_angles, circular_mean_deg,
                                 classifier_auc, evaluate_pitcher, fit_pitch_type, marginal_ks,
                                 movement_in, sample)
from pitchsim.physics import G, air_density
from pitchsim.statcast import _time_to_y, fit_plate_crossing, release_state
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


def _gauss(n, shift=0.0, seed=0):
    """Synthetic 8-D sample; shift moves release_speed by `shift` mph."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(rng.normal(0, 1, (n, 8)), columns=FEATURES)
    df["release_speed"] += 95 + shift; df["spin_axis"] = (200 + 5 * df["spin_axis"]) % 360
    return df


def test_auc_near_half_for_same_distribution():
    real = _gauss(300)
    auc, lo, hi = classifier_auc(real, [_gauss(300, seed=s) for s in range(1, 4)], GRADED, 200)
    assert auc < 0.6 and lo < 0.5 < hi


def test_auc_high_for_shifted_distribution():
    auc, _, _ = classifier_auc(_gauss(300), [_gauss(300, shift=2, seed=1)], GRADED, 200)
    assert auc > 0.8


def test_marginal_ks_flags_shifted_feature_only():
    ks = marginal_ks(_gauss(500), _gauss(500, shift=2, seed=1), 200).set_index("feature")
    assert ks.loc["release_speed", "flagged"] and not ks.loc["release_spin_rate", "flagged"]


def test_evaluate_pitcher_on_synthetic():
    """Synthetic pitcher, real-data path. Model is exact-ish, so most types pass."""
    df = make(n=1500, seed=0)
    summary, ks, skipped, clouds = evaluate_pitcher(df, np.full(len(df), air_density()))
    g = summary[summary.model == "gaussian"]
    assert (g.auc < 0.65).all() and set(g.pitch_type) == {"FF", "SL", "CH", "CU"}


def test_thin_types_skipped_or_ungraded():
    """Synthetic: trimmed arsenal."""
    df = make(n=1500, seed=0)
    cu = df.index[df.pitch_type == "CU"]
    df = df.drop(cu[20:])                            # 20 CU total -> < 30 train
    ch = df.index[df.pitch_type == "CH"]
    df = df.drop(ch[70:])                            # ~70 CH -> < 30 held-out
    summary, _, skipped, _ = evaluate_pitcher(df, np.full(len(df), air_density()))
    assert any(s.startswith("CU") for s in skipped)
    ch_rows = summary[summary.pitch_type == "CH"]
    assert (~ch_rows.graded).all() and ch_rows["pass"].isna().all()
    assert (ch_rows.n_components == 1).all()


def test_nan_extension_rows_dropped():
    df = make(n=1500, seed=0); df.loc[:9, "release_extension"] = np.nan
    _, _, skipped, _ = evaluate_pitcher(df, np.full(len(df), air_density()))
    assert any("10" in s and "release_extension" in s for s in skipped)


def test_movement_formula_matches_statcast_fit():
    """For a constant-acceleration path the formula returns 0.5*a*t^2 exactly."""
    df = make(n=50, seed=0)
    _, v_rel = release_state(df); _, v_plate, t_plate = fit_plate_crossing(df)
    t_rel = _time_to_y(df.release_pos_y.to_numpy(), df.vy0.to_numpy(), df.ay.to_numpy())
    t = t_plate - t_rel
    hb, vb = movement_in(v_rel, v_plate, t)
    assert np.allclose(hb, 0.5 * df.ax * t ** 2 * 12)
    assert np.allclose(vb, 0.5 * (df.az + G) * t ** 2 * 12)
