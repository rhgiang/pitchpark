"""M6 pitch-selection models vs the no-context mix, with grouped cross-validation by game.

Every game is held out exactly once (K folds of whole games), so each number is an average over
all of a pitcher's games, and the fold-to-fold spread shows whether a gain beats split noise.
All models see the same folds. Each model's one hyperparameter is chosen per fold on a 30% game
slice of that fold's TRAINING games (nested CV), never on its test games.
Writes results/m6_<model>.csv (per fold), m6_reliability.csv, m6_calibration.png.
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
from pitchsim.sequence.context import add_context
from pitchsim.sequence.logistic import FEATURES, BoostedSeq, LogisticSeq
from pitchsim.validation import game_folds, split_by_game

PITCHERS = {"Ohtani": 660271, "Misiorowski": 694819, "Sale": 519242, "Cease": 656302, "Yamamoto": 808967,
            "King": 650633, "Sanchez": 650911, "Schlittler": 693645}
# model name -> (class, hyperparameter grid)
MODELS = {"count": (lambda t, h: CountBaseline(t, a=h), [0.3, 1, 3, 10, 30, 100]),
          "logistic": (lambda t, h: LogisticSeq(t, C=h), [0.01, 0.03, 0.1, 0.3, 1, 3])}
# Boosted trees on the logistic's best feature set; hyperparameter = number of boosting rounds.
BEST = FEATURES + ["tto"]
MODELS["boosted"] = (lambda t, h: BoostedSeq(t, C=h, features=BEST), [25, 50, 100, 200, 400])
# Additions: the logistic model with one more context feature. Its loss drop is what that feature adds.
EXTRA = ["pitch_of_game", "tto", "outs", "runners", "score_diff"]
for add in EXTRA:
    MODELS[f"logistic_plus_{add}"] = (lambda t, h, a=add: LogisticSeq(t, C=h, features=FEATURES + [a]), [0.1, 0.3, 1, 3])
MODELS["logistic_fatigue"] = (lambda t, h: LogisticSeq(t, C=h, features=FEATURES + ["pitch_of_game", "tto"]), [0.1, 0.3, 1, 3])
MODELS["logistic_all"] = (lambda t, h: LogisticSeq(t, C=h, features=FEATURES + EXTRA), [0.03, 0.1, 0.3, 1, 3])
# Ablations: the logistic model with one feature removed. The loss it gains back over "logistic"
# is what that feature was worth. Printed, not plotted.
for drop in ["count", "stand", "prev_type"]:
    keep = [f for f in ["count", "stand", "prev_type"] if f != drop]
    MODELS[f"logistic_no_{drop}"] = (lambda t, h, k=keep: LogisticSeq(t, C=h, features=k), [0.03, 0.1, 0.3, 1, 3])
K = 5
OUT = Path(__file__).resolve().parents[1] / "results"
SEED = 0


def tune(make, grid, train: pd.DataFrame, types: list[str]) -> float:
    """Hyperparameter with the lowest log loss on a 30% game slice of the training games."""
    fit, val = split_by_game(train, seed=SEED)
    y = val["pitch_type"].map({t: i for i, t in enumerate(types)}).to_numpy()
    return min(grid, key=lambda h: log_loss(make(types, h).fit(fit).predict(val), y))


def main() -> None:
    rows, rel_rows, curves = [], [], {}
    for name, pid in PITCHERS.items():
        df = add_context(load_seasons(sorted(RAW_DIR.glob(f"{pid}_*.csv"))))
        df = df.dropna(subset=["pitch_type"])   # after add_context, so untyped pitches break chains
        types = sorted(df["pitch_type"].unique())
        idx = {t: i for i, t in enumerate(types)}
        held = {m: ([], []) for m in MODELS}
        for f, test_games in enumerate(game_folds(df, K, SEED)):
            test = df[df["game_pk"].isin(test_games)]
            train = df[~df["game_pk"].isin(test_games)]
            y = test["pitch_type"].map(idx).to_numpy()
            mix = train["pitch_type"].value_counts().reindex(types, fill_value=0).to_numpy() + 1.0
            row = {"pitcher": name, "fold": f, "test_games": len(test_games), "test_pitches": len(test),
                   "logloss_mix": log_loss(np.tile(mix / mix.sum(), (len(test), 1)), y)}
            for m, (make, grid) in MODELS.items():
                h = tune(make, grid, train, types)
                p = make(types, h).fit(train).predict(test)
                row[f"{m}_hparam"], row[f"logloss_{m}"] = h, log_loss(p, y)
                held[m][0].append(p); held[m][1].append(y)
            rows.append(row)
        for m, (ps, ys) in held.items():
            rel = reliability(np.vstack(ps), np.concatenate(ys))   # pooled over all held-out folds
            curves[(name, m)] = rel
            rel_rows.append(rel.assign(pitcher=name, model=m).reset_index())
    res = pd.DataFrame(rows)
    OUT.mkdir(exist_ok=True)
    res.to_csv(OUT / "m6_folds.csv", index=False)
    pd.concat(rel_rows).to_csv(OUT / "m6_reliability.csv", index=False)
    # pitch-weighted means so a small fold doesn't count as much as a big one
    wavg = lambda g, c: np.average(g[c], weights=g.test_pitches)
    summary = res.groupby("pitcher", sort=False).apply(lambda g: pd.Series({
        "pitches": g.test_pitches.sum(), "mix": wavg(g, "logloss_mix"),
        "count": wavg(g, "logloss_count"), "logistic": wavg(g, "logloss_logistic"),
        "gain_vs_count": wavg(g, "logloss_count") - wavg(g, "logloss_logistic"),
        "gain_sd_folds": (g.logloss_count - g.logloss_logistic).std(),
        "folds_beat_count": f"{(g.logloss_logistic < g.logloss_count).sum()}/{len(g)}",
        "C_chosen": ",".join(f"{x:g}" for x in g.logistic_hparam)}))
    print(summary.round(4).to_string())
    abl = [m for m in MODELS if m.startswith("logistic_no_")]
    worth = res.groupby("pitcher", sort=False).apply(lambda g: pd.Series(
        {m.removeprefix("logistic_no_"): wavg(g, f"logloss_{m}") - wavg(g, "logloss_logistic") for m in abl}))
    print("\nlog loss added by removing each feature from the logistic model (its worth):")
    print(worth.round(4).to_string())
    plus = [m for m in MODELS if m.startswith("logistic_plus_")] + ["logistic_fatigue", "logistic_all"]
    adds = res.groupby("pitcher", sort=False).apply(lambda g: pd.Series(
        {m.removeprefix("logistic_plus_"): wavg(g, "logloss_logistic") - wavg(g, f"logloss_{m}") for m in plus}))
    print("\nlog loss saved by adding each feature to the logistic model (+ = helps):")
    print(adds.round(4).to_string())
    wins = res.groupby("pitcher", sort=False).apply(lambda g: pd.Series(
        {m.removeprefix("logistic_plus_"): f"{(g[f'logloss_{m}'] < g.logloss_logistic).sum()}/{len(g)}" for m in plus}))
    print("folds where it beats logistic:"); print(wins.to_string())
    vs = res.groupby("pitcher", sort=False).apply(lambda g: pd.Series({
        "logistic+tto": wavg(g, "logloss_logistic_plus_tto"), "boosted": wavg(g, "logloss_boosted"),
        "gain": wavg(g, "logloss_logistic_plus_tto") - wavg(g, "logloss_boosted"),
        "folds_won": f"{(g.logloss_boosted < g.logloss_logistic_plus_tto).sum()}/{len(g)}",
        "rounds": ",".join(f"{x:g}" for x in g.boosted_hparam)}))
    print("\nboosted trees vs logistic+tto (same features):"); print(vs.round(4).to_string())
    big = pd.concat(rel_rows).query("n >= 100 and model in ['count', 'logistic', 'boosted']")
    print("\nlargest calibration gap in bins with n >= 100:")
    print((big.observed - big.mean_pred).abs().groupby([big.pitcher, big.model]).max().round(3).to_string())
    shown = ["count", "logistic", "boosted"]
    fig, axes = plt.subplots(1, len(shown), figsize=(4.5 * len(shown), 4.5), sharey=True)
    for ax, m in zip(axes, shown):
        ax.plot([0, 1], [0, 1], "k:", lw=1)
        for name in PITCHERS:
            r = curves[(name, m)].query("n >= 30")   # tiny bins are noise
            ax.plot(r.mean_pred, r.observed, "o-", label=name)
        ax.set(xlabel="predicted probability", title=f"{m} ({K}-fold by game, bins n>=30)")
    axes[0].set_ylabel("observed frequency"); axes[0].legend()
    fig.tight_layout(); fig.savefig(OUT / "m6_calibration.png", dpi=120)


if __name__ == "__main__":
    sys.exit(main())
