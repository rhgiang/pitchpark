"""Viewer export (M4). All data here is SYNTHETIC (pitchsim.synthetic.make plus made-up at-bat
columns); nothing here is a real pitch.

    python tests/test_export.py     # regenerates web/test/fixtures/synthetic.json
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pitchsim.export import export_pitcher, write_json
from pitchsim.generative import MIN_TRAIN
from pitchsim.physics import FRONT_OF_PLATE_Y, air_density
from pitchsim.seam import fit_pitch_types, predict_push
from pitchsim.statcast import clean
from pitchsim.synthetic import make
from pitchsim.validation import run_sim, split_by_game

FIXTURE = Path(__file__).resolve().parents[1] / "web" / "test" / "fixtures" / "synthetic.json"
THIN = "SL"         # thinned below MIN_TRAIN so it has no M3 model
PER_AB = 4          # pitches per synthetic at-bat
BALLS = [0, 1, 1, 2]
STRIKES = [0, 0, 1, 1]


def synthetic_raw(n: int = 700, seed: int = 3) -> tuple[pd.DataFrame, tuple[int, int]]:
    """SYNTHETIC raw Statcast-like rows with at-bat columns, shuffled.

    Returns (raw, (game_pk, at_bat_number)) of a held-out at-bat whose pitch 2 has NaN
    spin_axis (no tracking). THIN keeps only 25 pitches, so it has < MIN_TRAIN training rows.
    """
    df = make(n=n, seed=seed)
    thin = np.flatnonzero(df["pitch_type"] == THIN)[25:]
    df = df.drop(index=thin).sort_values("game_pk", kind="stable").reset_index(drop=True)
    k = df.groupby("game_pk").cumcount().to_numpy()
    ab = k // PER_AB + 1
    pn = k % PER_AB + 1
    last = (pn == PER_AB) | (df["game_pk"] != df["game_pk"].shift(-1)).to_numpy()
    df = df.assign(
        at_bat_number=ab, pitch_number=pn,
        balls=[BALLS[i - 1] for i in pn], strikes=[STRIKES[i - 1] for i in pn],
        game_date=[f"2026-05-{g % 28 + 1:02d}" for g in df["game_pk"]],
        inning=(ab - 1) // 3 + 1, inning_topbot="Top", outs_when_up=(ab - 1) % 3,
        stand=np.where(ab % 2, "R", "L"), batter=100000 + ab, pitcher=1, p_throws="R",
        description=np.where(last, "hit_into_play", "ball"),
        events=np.where(last, "field_out", None),
        sz_top=3.4, sz_bot=1.6, arm_angle=40.0)

    held = set(split_by_game(clean(df), seed=0)[1]["game_pk"])
    g = sorted(held)[0]
    row = df.index[(df["game_pk"] == g) & (df["at_bat_number"] == 1) & (df["pitch_number"] == 2)]
    df.loc[row, "spin_axis"] = np.nan
    assert set(split_by_game(clean(df), seed=0)[1]["game_pk"]) == held  # split unchanged
    return df.sample(frac=1, random_state=0).reset_index(drop=True), (g, 1)


@pytest.fixture(scope="module")
def exported():
    raw, ab = synthetic_raw()
    rho = pd.Series(air_density(), index=raw.index)
    return raw, ab, export_pitcher(raw, rho)


def _pitches(out):
    for a in out["at_bats"]:
        for p in a["pitches"]:
            yield a, p


def test_only_held_out_games(exported):
    raw, _, out = exported
    held = set(split_by_game(clean(raw), seed=0)[1]["game_pk"])
    games = {a["game_pk"] for a in out["at_bats"]}
    assert games and games <= held


def test_pitch_order_and_count_from_raw(exported):
    _, (g, n), out = exported
    ab = next(a for a in out["at_bats"] if a["game_pk"] == g and a["id"] == f"{g}-{n}")
    ps = ab["pitches"]
    assert [p["pitch_number"] for p in ps] == [1, 2, 3, 4]
    assert [p["balls"] for p in ps] == BALLS and [p["strikes"] for p in ps] == STRIKES
    assert ps[1]["status"] == "no_tracking"
    assert ps[1]["real"] is None and ps[1]["sim"] is None
    assert ps[2]["tunnel"]["sep_prev_in"] is None    # previous pitch has no path


def test_no_model_type(exported):
    _, _, out = exported
    thin = [p for _, p in _pitches(out) if p["pitch_type"] == THIN]
    assert thin and MIN_TRAIN > 25
    for p in thin:
        if p["status"] != "no_tracking":
            assert p["status"] == "no_model" and p["real"] is not None and p["sim"] is None
    assert out["pitch_types"][THIN]["modelled"] is False
    assert out["pitch_types"]["FF"]["modelled"] is True


def test_paths_end_at_plate(exported):
    _, _, out = exported
    for _, p in _pitches(out):
        if p["real"] is None:
            continue
        x, y, z = p["real"]["p"][-1]
        assert abs(x - p["plate_real"][0]) <= 1e-4 and abs(z - p["plate_real"][1]) <= 1e-4
        assert p["real"]["t"][-1] == p["plate_real"][2] and p["real"]["t"][0] == 0
        assert abs(y - FRONT_OF_PLATE_Y) <= 1e-4
        if p["sim"] is not None:
            assert abs(p["sim"]["p"][-1][1] - FRONT_OF_PLATE_Y) <= 1e-4


def test_sim_matches_m3(exported):
    raw, _, out = exported
    df = clean(raw.assign(rho=air_density()))
    _, test, fits, _ = fit_pitch_types(df, df["rho"])
    key = test.set_index(["game_pk", "at_bat_number", "pitch_number"]).index
    n = 0
    for a, p in _pitches(out):
        if p["status"] != "ok":
            continue
        i = key.get_loc((a["game_pk"], int(a["id"].split("-")[1]), p["pitch_number"]))
        rows = test.iloc[[i]]
        tf = fits[p["pitch_type"]]
        pos, _, t = run_sim(rows, tf.cd, tf.eff, "release", predict_push(tf.per, rows))
        assert np.allclose(p["plate_sim"], [pos[0, 0], pos[0, 2], t[0]], atol=1e-4)
        n += 1
    assert n > 50


def test_synthetic_flag(exported):
    _, _, out = exported
    assert out["meta"]["synthetic"] is True and "Synthetic" in out["pitcher"]["name"]


def write_fixture() -> None:
    """Two SYNTHETIC at-bats for the web tests: the no-tracking one and one with a no-model
    pitch; together they include a CU."""
    raw, (g, n) = synthetic_raw()
    out = export_pitcher(raw, pd.Series(air_density(), index=raw.index))
    first = next(a for a in out["at_bats"] if a["id"] == f"{g}-{n}")
    second = next(a for a in out["at_bats"] if a is not first
                  and any(p["status"] == "no_model" for p in a["pitches"])
                  and any(p["pitch_type"] == "CU" for p in a["pitches"] + first["pitches"]))
    out["at_bats"] = [first, second]
    used = {p["pitch_type"] for a in out["at_bats"] for p in a["pitches"]}
    out["pitch_types"] = {k: v for k, v in out["pitch_types"].items() if k in used}
    write_json(out, FIXTURE)
    print(f"wrote {FIXTURE}")


if __name__ == "__main__":
    write_fixture()
