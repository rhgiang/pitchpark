import numpy as np
import pytest

from pitchsim.physics import air_density
from pitchsim.synthetic import make
from pitchsim.validation import evaluate, split_by_game


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
    print(rows[["pitch_type", "cd", "calibrated_err_in"]])
    assert rows["cd"].between(0.33, 0.37).all()
    assert (rows["calibrated_err_in"] < 1.0).all()
    assert (rows["n_test"] > 0).all()
    assert (rows["n_train"] + rows["n_test"] <= rows["n"]).all()


def test_evaluate_lists_rare_pitch_types_as_skipped(big):
    df = big.copy()
    df = df[(df.pitch_type != "CU") | (df.index < df[df.pitch_type == "CU"].index[10])]
    _, skipped = evaluate(df, "release", np.full(len(df), air_density()))
    assert any(s.startswith("CU") for s in skipped)
