"""M6 count baseline vs the no-context mix, with grouped cross-validation by game.

Every game is held out exactly once (K folds of whole games), so each number is an average over
all of a pitcher's games instead of one 8-13 game test set, and the fold-to-fold spread shows
whether a gain is bigger than split noise. The smoothing strength `a` is chosen per fold on a
validation slice of that fold's TRAINING games (nested CV), never on its test games.
Writes results/m6_baseline.csv, m6_baseline_reliability.csv, m6_baseline_calibration.png.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pitchsim.data import RAW_DIR, load_seasons
from pitchsim.sequence.baseline import CountBaseline, log_loss, reliability
from pitchsim.validation import split_by_game

PITCHERS = {"Ohtani": 660271, "Misiorowski": 694819, "Sale": 519242}
A_GRID = [0.3, 1, 3, 10, 30, 100]
K = 5
OUT = Path(__file__).resolve().parents[1] / "results"
SEED = 0


def game_folds(df: pd.DataFrame, k: int, seed: int) -> list[np.ndarray]:
    """K disjoint sets of game_pks covering every game once."""
    games = np.random.default_rng(seed).permutation(df["game_pk"].unique())
    return np.array_split(games, k)


def pick_a(train: pd.DataFrame, types: list[str]) -> float:
    """Smoothing strength with the lowest log loss on a 30% game slice of the training games."""
    fit, val = split_by_game(train, seed=SEED)
    y = val["pitch_type"].map({t: i for i, t in enumerate(types)}).to_numpy()
    return min(A_GRID, key=lambda a: log_loss(CountBaseline(types, a).fit(fit).predict(val), y))


def main() -> None:
    rows, rel_rows, curves = [], [], {}
    for name, pid in PITCHERS.items():
        df = load_seasons(sorted(RAW_DIR.glob(f"{pid}_*.csv")))
        df = df.dropna(subset=["pitch_type", "balls", "strikes"])
        types = sorted(df["pitch_type"].unique())
        idx = {t: i for i, t in enumerate(types)}
        ps, ys = [], []
        for f, test_games in enumerate(game_folds(df, K, SEED)):
            test = df[df["game_pk"].isin(test_games)]
            train = df[~df["game_pk"].isin(test_games)]
            y = test["pitch_type"].map(idx).to_numpy()
            a = pick_a(train, types)
            model = CountBaseline(types, a).fit(train)
            p = model.predict(test)
            flat = np.tile(model.overall, (len(test), 1))   # reference: ignore the count entirely
            rows.append({"pitcher": name, "fold": f, "a": a, "test_games": len(test_games),
                         "test_pitches": len(test), "logloss_overall": log_loss(flat, y),
                         "logloss_count": log_loss(p, y)})
            ps.append(p); ys.append(y)
        rel = reliability(np.vstack(ps), np.concatenate(ys))   # pooled over all held-out folds
        curves[name] = rel
        rel_rows.append(rel.assign(pitcher=name).reset_index())
    res = pd.DataFrame(rows).assign(gain=lambda d: d.logloss_overall - d.logloss_count)
    OUT.mkdir(exist_ok=True)
    res.to_csv(OUT / "m6_baseline.csv", index=False)
    pd.concat(rel_rows).to_csv(OUT / "m6_baseline_reliability.csv", index=False)
    # pitch-weighted means so a small fold doesn't count as much as a big one
    w = lambda c: (lambda g: np.average(g[c], weights=g.test_pitches))
    summary = res.groupby("pitcher", sort=False).apply(lambda g: pd.Series({
        "games": g.test_games.sum(), "pitches": g.test_pitches.sum(),
        "a_chosen": ",".join(f"{x:g}" for x in g.a),
        "logloss_overall": w("logloss_overall")(g), "logloss_count": w("logloss_count")(g),
        "gain": w("gain")(g), "gain_sd_folds": g.gain.std(), "folds_improved": f"{(g.gain > 0).sum()}/{len(g)}"}))
    print(summary.round(4).to_string())
    print(pd.concat(rel_rows).round(3).to_string(index=False))
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot([0, 1], [0, 1], "k:", lw=1)
    for name, r in curves.items():
        ax.plot(r.mean_pred, r.observed, "o-", label=name)
    ax.set(xlabel="predicted probability", ylabel="observed frequency",
           title=f"Count baseline calibration ({K}-fold by game)")
    ax.legend(); fig.tight_layout(); fig.savefig(OUT / "m6_baseline_calibration.png", dpi=120)


if __name__ == "__main__":
    sys.exit(main())
