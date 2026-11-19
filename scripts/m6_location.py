"""M6 target location: context-conditioned 2D Gaussian vs unconditional Gaussian, grouped CV by game.

Same 5 game folds as m6_evaluate.py. Target = where Statcast's own 9-parameter fit crosses the
FRONT of the plate (y = 17/12 ft), not plate_x/plate_z (mid-plate in the 2026 CSVs, see CLAUDE.md).
Ridge strength is chosen per fold on a 30% game slice of that fold's TRAINING games, never on test.
Writes results/m6_location_folds.csv and results/m6_location_types.csv.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from pitchsim.data import RAW_DIR, load_seasons
from pitchsim.sequence.context import add_context
from pitchsim.sequence.location import LocationModel
from pitchsim.statcast import fit_plate_crossing
from pitchsim.validation import game_folds, split_by_game

PITCHERS = {"Ohtani": 660271, "Misiorowski": 694819, "Sale": 519242}
GRID = [1, 3, 10, 30, 100, 300]
MIN_TRAIN = 30   # pitches of one type in a fold's training games
K, SEED = 5, 0
OUT = Path(__file__).resolve().parents[1] / "results"
FIT_COLS = ["vx0", "vy0", "vz0", "ax", "ay", "az", "release_pos_x", "release_pos_y", "release_pos_z"]


def add_location(df: pd.DataFrame) -> pd.DataFrame:
    """x_ft, z_ft = the fit's crossing of the front of the plate; NaN where the fit is incomplete."""
    ok = df[FIT_COLS].notna().all(axis=1).to_numpy()
    d = df.assign(x_ft=np.nan, z_ft=np.nan)
    pos, _, _ = fit_plate_crossing(df[ok])
    d.loc[ok, "x_ft"], d.loc[ok, "z_ft"] = pos[:, 0], pos[:, 2]
    return d


def per_type(make, train: pd.DataFrame, test: pd.DataFrame):
    """Fit one model per pitch type; yield (type, model, test rows of that type)."""
    for t, tr in train.groupby("pitch_type"):
        te = test[test["pitch_type"] == t]
        if len(tr) >= MIN_TRAIN and len(te):
            yield t, make().fit(tr), te


def tune(train: pd.DataFrame) -> float:
    """Ridge alpha with the best total log-likelihood on a 30% game slice of the training games."""
    fit, val = split_by_game(train, seed=SEED)
    def score(a):
        return sum(m.logpdf(te).sum() for _, m, te in per_type(lambda: LocationModel(alpha=a), fit, val))
    return max(GRID, key=score)


def main() -> None:
    fold_rows, type_rows = [], []
    for name, pid in PITCHERS.items():
        df = add_context(load_seasons(sorted(RAW_DIR.glob(f"{pid}_*.csv"))))
        df = add_location(df.dropna(subset=["pitch_type"]))   # same preprocessing as m6_evaluate => same folds
        for f, test_games in enumerate(game_folds(df.dropna(subset=["pitch_type"]), K, SEED)):
            held = df["game_pk"].isin(test_games)
            train, test = df[~held].dropna(subset=["x_ft"]), df[held].dropna(subset=["x_ft"])
            a = tune(train)
            ll_ref, ll_cond, n = 0.0, 0.0, 0
            for t, m, te in per_type(lambda: LocationModel(alpha=a), train, test):
                ref = LocationModel(features=[]).fit(train[train["pitch_type"] == t])
                ll_ref += ref.logpdf(te).sum(); ll_cond += m.logpdf(te).sum(); n += len(te)
                err = np.hypot(*(te[["x_ft", "z_ft"]].to_numpy() - m.mean(te)).T)
                type_rows.append({"pitcher": name, "fold": f, "pitch_type": t, "n": len(te),
                                  "err_ft": err, "in50": m.inside(te, 0.5).sum(), "in90": m.inside(te, 0.9).sum()})
            fold_rows.append({"pitcher": name, "fold": f, "alpha": a, "n_scored": n,
                              "n_dropped_rare_type": len(test) - n, "n_dropped_no_location": int(held.sum() - len(test)),
                              "ll_ref": ll_ref / n, "ll_cond": ll_cond / n})
    folds = pd.DataFrame(fold_rows)
    OUT.mkdir(exist_ok=True)
    folds.to_csv(OUT / "m6_location_folds.csv", index=False)
    wavg = lambda g, c: np.average(g[c], weights=g.n_scored)
    print("held-out mean log-likelihood per pitch (nats per ft^2, higher is better):")
    print(folds.groupby("pitcher", sort=False).apply(lambda g: pd.Series({
        "pitches": g.n_scored.sum(), "dropped_rare": g.n_dropped_rare_type.sum(),
        "dropped_no_loc": g.n_dropped_no_location.sum(), "reference": wavg(g, "ll_ref"),
        "conditioned": wavg(g, "ll_cond"), "gain": wavg(g, "ll_cond") - wavg(g, "ll_ref"),
        "gain_sd_folds": (g.ll_cond - g.ll_ref).std(), "folds_beat_ref": f"{(g.ll_cond > g.ll_ref).sum()}/{len(g)}",
        "alpha": ",".join(f"{x:g}" for x in g.alpha)})).round(4).to_string())
    t = pd.DataFrame(type_rows)
    # median over pitches: expand each row's err (a per-pitch array) -- stored as arrays above
    err = t.explode("err_ft").astype({"err_ft": float}).groupby(["pitcher", "pitch_type"], sort=False)["err_ft"].median()
    g = t.groupby(["pitcher", "pitch_type"], sort=False)[["n", "in50", "in90"]].sum()
    g["median_err_in"] = err * 12
    g["cover50"], g["cover90"] = g.in50 / g.n, g.in90 / g.n
    g.reset_index().to_csv(OUT / "m6_location_types.csv", index=False)
    print("\nper pitch type (median miss of model mean in inches; coverage of central 50% / 90% ellipses):")
    print(g[["n", "median_err_in", "cover50", "cover90"]].round(3).to_string())
    for p, gg in g.groupby(level=0, sort=False):
        print(f"{p}: pooled coverage 50% = {gg.in50.sum() / gg.n.sum():.3f}, 90% = {gg.in90.sum() / gg.n.sum():.3f}")


if __name__ == "__main__":
    sys.exit(main())
