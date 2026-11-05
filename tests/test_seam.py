import numpy as np
import pandas as pd
import pytest

from pitchsim.physics import air_density
from pitchsim.seam import (FORBIDDEN, SEAM_FEATURES, fit_fixed, fit_per_pitch,
                           predict_push, push_target, seam_design)
from pitchsim.statcast import fit_plate_crossing
from pitchsim.synthetic import make
from pitchsim.validation import residuals


def _data(n, seed, push):
    """SYNTHETIC pitches with a known push, true Cd / spin efficiency, and rho."""
    df = make(n=n, seed=seed, push=push)
    df["rho"] = air_density()
    return df


@pytest.fixture(scope="module")
def const():
    df = _data(300, 1, lambda c, a: (4.0, -3.0))
    target = push_target(df, df["true_cd"].to_numpy(), df["true_spin_eff"].to_numpy())
    # Same seed with zero push: make's rng draws don't depend on push, so these are the same
    # pitches. Its target is pure release-state artefact (see seam.py), differenced out below.
    zero = _data(300, 1, None)
    base = push_target(zero, zero["true_cd"].to_numpy(), zero["true_spin_eff"].to_numpy())
    return df, target, base


def test_push_target_closes_the_miss(const):
    """SYNTHETIC: re-simulating with the target leaves ~no miss at the fit crossing."""
    df, target, base = const
    fit_pos, fit_vel, _ = fit_plate_crossing(df)
    dx, dz, _ = residuals(df, fit_pos, fit_vel, df["true_cd"].to_numpy(),
                          df["true_spin_eff"].to_numpy(), "release", extra_acc=target)
    assert np.median(np.hypot(dx, dz)) < 0.05
    _check_recovers(df, target, base)


def _check_recovers(df, target, base):
    """Mean of (target - zero-push target) is (4, -3) overall and within each pitch type."""
    diff = target - base
    assert np.allclose(diff.mean(axis=0), [4.0, -3.0], atol=0.2)
    for code in df["pitch_type"].unique():
        m = (df["pitch_type"] == code).to_numpy()
        assert np.allclose(diff[m].mean(axis=0), [4.0, -3.0], atol=0.2), code


def test_fixed_recovers_constant_push(const):
    """SYNTHETIC: fit_fixed's mean tracks the constant push once the artefact is differenced out."""
    df, target, base = const
    _check_recovers(df, target, base)
    assert np.allclose(fit_fixed(target).mean - fit_fixed(base).mean, [4.0, -3.0], atol=0.2)


def test_per_pitch_recovers_axis_slope():
    """SYNTHETIC: push_x rises with spin axis for CH only; ridge finds the slope."""
    push = lambda c, a: (0.15 * (a - 240), 0.0) if c == "CH" else (0.0, 0.0)
    train = _data(600, 1, push)
    train = train[train.pitch_type == "CH"].reset_index(drop=True)
    target = push_target(train, train["true_cd"].to_numpy(), train["true_spin_eff"].to_numpy())
    model = fit_per_pitch(train, target)
    fresh = _data(600, 2, push)
    fresh = fresh[fresh.pitch_type == "CH"]
    pred = predict_push(model, fresh)
    assert np.corrcoef(pred[:, 0], fresh["true_push_x"])[0, 1] > 0.9


def test_design_has_no_jump_at_zero():
    df = make(n=2, seed=0)
    df.loc[:, :] = df.iloc[[0, 0]].values
    df["spin_axis"] = [359.5, 0.5]
    d = seam_design(df)
    assert d.shape == (2, len(SEAM_FEATURES))
    assert np.abs(d[0] - d[1]).max() < 0.02


def test_design_never_reads_movement():
    df = make(n=5, seed=0)
    df[FORBIDDEN] = np.nan
    assert np.isfinite(seam_design(df)).all()


def test_few_games_still_fits():
    """SYNTHETIC: 3 games fit a ridge; 1 game falls back to the fixed mean."""
    df = make(n=40, seed=3, push=lambda c, a: (1.0, 0.0))
    df["game_pk"] = np.arange(40) % 3
    target = np.column_stack([np.full(40, 1.0), np.zeros(40)])
    assert fit_per_pitch(df, target).kind == "per_pitch"
    df["game_pk"] = 7
    assert fit_per_pitch(df, target).kind == "fixed_fallback"


# ---- evaluation (Task 4) ----
from pitchsim.seam import MIN_DROP_IN, evaluate_pitcher, grade, median_drop_ci  # noqa: E402

SUMMARY_COLS = ["pitch_type", "pitch_name", "n_train", "n_test", "n_games_test",
                "n_games_train", "cd", "spin_eff",
                "err_m1_in", "err_fixed_in", "err_per_pitch_in", "bias_x_m1_in", "bias_z_m1_in",
                "bias_x_m3_in", "bias_z_m3_in", "drop_in", "drop_lo_in", "drop_hi_in",
                "pp_vs_fixed_in", "pp_vs_fixed_lo_in", "pp_vs_fixed_hi_in", "push_x_fts2",
                "push_z_fts2", "axis_gap_deg", "alpha", "model_kind", "graded", "pass"]
# Off-axis on purpose: a push along the spin force (e.g. (5, -4) here) is absorbed whole by M1's
# spin-efficiency calibration, leaving no miss for M3. A seam push is what spin can't explain.
_CH_PUSH = lambda c, a: (4.0, 5.0) if c == "CH" else (0.0, 0.0)


def test_median_drop_ci_hand_values():
    games = np.repeat(np.arange(10), 4)
    assert median_drop_ci(np.full(40, 5.0), np.full(40, 3.0), games) == (2, 2, 2)
    x = np.arange(40.0)
    assert median_drop_ci(x, x, games) == (0, 0, 0)


@pytest.fixture(scope="module")
def evaluated():
    df = make(n=1500, seed=0, push=_CH_PUSH)
    rho = np.full(len(df), air_density())
    return evaluate_pitcher(df, rho)


def test_evaluate_on_synthetic_push(evaluated):
    summary, _, _ = evaluated
    assert list(summary.columns) == SUMMARY_COLS
    ch = summary[summary.pitch_type == "CH"].iloc[0]
    assert ch["err_per_pitch_in"] < ch["err_m1_in"]
    assert ch["pass"] is True or ch["pass"] == True  # noqa: E712


def test_no_seam_types_do_not_pass(evaluated):
    """Synthetic: only CH has a seam push, so the constant M1 offset alone must not pass."""
    summary, _, _ = evaluated
    others = summary[(summary.pitch_type != "CH") & summary.graded]
    assert len(others) > 0
    assert not others["pass"].astype(bool).any()


def test_grade_rule():
    assert grade(100, 10, (MIN_DROP_IN, 0.1, 1.0)) == (True, True)
    assert grade(100, 10, (MIN_DROP_IN - 0.01, 0.1, 1.0)) == (True, False)  # too small
    assert grade(100, 10, (2.0, -0.1, 3.0)) == (True, False)  # interval touches zero
    assert grade(29, 10, (2.0, 1.0, 3.0)) == (False, pd.NA)  # too few pitches
    assert grade(100, 1, (2.0, 2.0, 2.0)) == (False, pd.NA)  # one game: no real interval


def test_evaluate_no_push_no_harm():
    df = make(n=1500, seed=0)
    summary, _, _ = evaluate_pitcher(df, np.full(len(df), air_density()))
    g = summary[summary.graded]
    assert len(g) > 0
    assert (g.err_per_pitch_in <= g.err_m1_in + 0.3).all()


def test_ungraded_type_never_passes():
    df = make(n=1500, seed=0, push=_CH_PUSH)
    ch = np.flatnonzero(df.pitch_type == "CH")
    df = df.drop(index=ch[70:]).reset_index(drop=True)
    summary, _, _ = evaluate_pitcher(df, np.full(len(df), air_density()))
    row = summary[summary.pitch_type == "CH"]
    if len(row):  # may be skipped outright if too few training pitches
        assert not row.iloc[0]["graded"]
        assert row.iloc[0]["pass"] is pd.NA


def test_nan_feature_rows_dropped():
    df = make(n=600, seed=0)
    df.loc[df.index[:10], "release_extension"] = np.nan
    _, _, skipped = evaluate_pitcher(df, np.full(len(df), air_density()))
    assert any("10" in s and "release_extension" in s for s in skipped)


def test_knock_on_shape(evaluated):
    summary, knock, _ = evaluated
    assert list(knock.columns) == ["pitch_type", "variant", "component", "ks_stat", "p_value",
                                   "mean_diff_in"]
    for t in summary.pitch_type:
        sub = knock[knock.pitch_type == t]
        assert set(zip(sub.variant, sub.component)) == {(v, c) for v in ("m1", "m1+m3")
                                                         for c in ("x", "z")}
