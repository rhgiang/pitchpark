import numpy as np
import pytest

from pitchsim.physics import air_density
from pitchsim.synthetic import make
from pitchsim.statcast import _time_to_y, fit_plate_crossing, initial_state
from pitchsim.validation import evaluate, residuals, split_by_game


@pytest.fixture(scope="module")
def big():
    return make(n=1500, seed=0)   # slow to build, so share it


def test_split_by_game_has_no_shared_games():
    df = make(n=600, seed=0)
    train, test = split_by_game(df)
    assert set(train.game_pk).isdisjoint(test.game_pk)
    assert len(train) + len(test) == len(df)


def test_split_by_game_is_deterministic():
    df = make(n=600, seed=0)
    assert split_by_game(df)[1].index.equals(split_by_game(df)[1].index)


def test_evaluate_recovers_true_cd_from_release(big):
    df = big.copy()
    rows, skipped = evaluate(df, "release", np.full(len(df), air_density()))
    assert rows["cd"].between(0.33, 0.37).all()
    assert (rows["calibrated_err_in"] < 1.0).all()
    assert (rows["n_test"] > 0).all()
    assert (rows["n_train"] + rows["n_test"] <= rows["n"]).all()


def test_evaluate_lists_rare_pitch_types_as_skipped(big):
    df = big.copy()
    df = df[(df.pitch_type != "CU") | (df.index < df[df.pitch_type == "CU"].index[10])]
    _, skipped = evaluate(df, "release", np.full(len(df), air_density()))
    assert any(s.startswith("CU") for s in skipped)


def test_residuals_ignore_where_plate_x_z_are_reported():
    """SYNTHETIC. 2026 Statcast reports plate_x/z at y ~ 0.708 ft, not 17/12. Scoring
    against the fit's own crossing must give the same residuals either way."""
    df = make(n=200, seed=1).assign(rho=air_density())
    pos0, vel0 = initial_state(df)
    acc = df[["ax", "ay", "az"]].to_numpy()
    t = _time_to_y(0.708, vel0[:, 1], acc[:, 1])
    moved = pos0 + vel0 * t[:, None] + 0.5 * acc * t[:, None] ** 2
    shifted = df.assign(plate_x=moved[:, 0], plate_z=moved[:, 2])
    assert np.median(np.abs(shifted.plate_z - df.plate_z)) * 12 > 0.5   # really moved

    def medians(d):
        fit_pos, fit_vel, _ = fit_plate_crossing(d)
        return [np.median(r) for r in residuals(d, fit_pos, fit_vel, 0.35, 0.8, "release")]

    assert medians(shifted) == medians(df)
