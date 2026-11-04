import numpy as np
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
    return df, target


def test_push_target_closes_the_miss(const):
    """SYNTHETIC: re-simulating with the target leaves ~no miss at the fit crossing."""
    df, target = const
    fit_pos, fit_vel, _ = fit_plate_crossing(df)
    dx, dz, _ = residuals(df, fit_pos, fit_vel, df["true_cd"].to_numpy(),
                          df["true_spin_eff"].to_numpy(), "release", extra_acc=target)
    assert np.median(np.hypot(dx, dz)) < 0.05
    # Not atol 0.2: the target closes the miss to Statcast's *fitted* crossing, and that
    # 9-parameter constant-acceleration fit sits ~0.3-0.6 in off the true path, i.e. ~0.3 ft/s^2.
    assert np.allclose(target.mean(axis=0), [4.0, -3.0], atol=0.4)


def test_fixed_recovers_constant_push(const):
    """SYNTHETIC: the column means of the target are the constant push (same fit-error slack)."""
    assert np.allclose(fit_fixed(const[1]).mean, [4.0, -3.0], atol=0.4)


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
