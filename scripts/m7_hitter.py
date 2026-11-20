"""M7 average hitter: HitterModel (logistic) vs HitterTable (count x zone bin), grouped CV by game.

Same 5 game folds as m6_location.py. C and a are chosen per fold on a 30% game slice of that
fold's TRAINING games, never on test. Writes results/m7/hitter.csv (stays local).
`use` = "model" if its held-out log loss beats the table's for that pitcher and stand, else "table".
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from m6_location import PITCHERS, add_location, K, SEED
from pitchsim.data import RAW_DIR, load_seasons
from pitchsim.sequence.context import add_context
from pitchsim.sequence.hitter import HitterModel, HitterTable, _y, prepare
from pitchsim.validation import game_folds, split_by_game

OUT = Path(__file__).resolve().parents[1] / "results" / "m7"
C_GRID, A_GRID = [0.03, 0.1, 0.3, 1, 3, 10], [0.3, 1, 3, 10, 30]


def nll(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per-pitch -log p[observed outcome]."""
    return -np.log(np.maximum(p[np.arange(len(y)), y], 1e-12))


def tune(train: pd.DataFrame) -> tuple[float, float]:
    fit, val = split_by_game(train, seed=SEED)
    val = prepare(val); y = _y(val)
    c = min(C_GRID, key=lambda h: nll(HitterModel(h).fit(fit).predict(val), y).mean())
    a = min(A_GRID, key=lambda h: nll(HitterTable(h).fit(fit).predict(val), y).mean())
    return c, a


def main() -> None:
    rows = []
    for name, pid in PITCHERS.items():
        df = add_context(load_seasons(sorted(RAW_DIR.glob(f"{pid}_*.csv"))))
        df = add_location(df.dropna(subset=["pitch_type"]))
        n_drop = len(df) - len(prepare(df))
        print(f"{name}: {len(df)} pitches, {n_drop} dropped (no outcome/location/zone)")
        for f, test_games in enumerate(game_folds(df, K, SEED)):
            held = df["game_pk"].isin(test_games)
            train, test = df[~held], prepare(df[held])
            c, a = tune(train)
            y = _y(test)
            l_m = nll(HitterModel(c).fit(train).predict(test), y)
            l_t = nll(HitterTable(a).fit(train).predict(test), y)
            for s, g in test.assign(l_m=l_m, l_t=l_t).groupby("stand"):
                rows.append({"pitcher": name, "stand": s, "fold": f, "n": len(g), "l_m": g["l_m"].sum(), "l_t": g["l_t"].sum()})
    r = pd.DataFrame(rows)
    out = r.groupby(["pitcher", "stand"], sort=False).apply(lambda g: pd.Series({
        "n_test": g.n.sum(), "ll_model": g["l_m"].sum() / g.n.sum(), "ll_table": g["l_t"].sum() / g.n.sum(),
        "folds_better": int((g["l_m"] / g.n < g["l_t"] / g.n).sum())}), include_groups=False).reset_index()
    out["n_test"], out["folds_better"] = out.n_test.astype(int), out.folds_better.astype(int)
    out["use"] = np.where(out.ll_model < out.ll_table, "model", "table")
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT / "hitter.csv", index=False)
    tot = r.groupby("pitcher", sort=False)[["n", "l_m", "l_t"]].sum()
    print("\noverall held-out log loss (nats/pitch):")
    print(pd.DataFrame({"model": tot.l_m / tot.n, "table": tot.l_t / tot.n}).round(4).to_string())
    print("\nper stand:\n" + out.round(4).to_string(index=False))


if __name__ == "__main__":
    sys.exit(main())
