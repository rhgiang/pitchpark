"""M6 transformer vs logistic, on the same 5 game folds as m6_evaluate.py.

The transformer is pooled: fold f trains one model on every pitcher's non-fold-f games and scores
each pitcher on his fold-f games. Two logistic references, each per pitcher with C tuned the same
nested way as in m6_evaluate:
  logistic+tto   the current default
  logistic+prev  + previous pitch's result and zone, the same inputs the transformer gets, so a
                 transformer gain over it comes from the longer history / pooling, not the inputs.
Writes results/m6_transformer_folds.csv, m6_transformer_reliability.csv, m6_transformer_calibration.png.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from m6_evaluate import K, OUT, PITCHERS, SEED, tune
from pitchsim.data import RAW_DIR, load_seasons
from pitchsim.sequence.baseline import log_loss, reliability
from pitchsim.sequence.context import add_context
from pitchsim.sequence.logistic import FEATURES, LogisticSeq
from pitchsim.sequence.transformer import TransformerSeq
from pitchsim.validation import game_folds

C_GRID = [0.03, 0.1, 0.3, 1, 3]
REFS = {"logistic+tto": FEATURES + ["tto"],
        "logistic+prev": FEATURES + ["tto", "prev_result", "prev_zone"]}


def main() -> None:
    data, folds = {}, {}
    for name, pid in PITCHERS.items():
        df = add_context(load_seasons(sorted(RAW_DIR.glob(f"{pid}_*.csv"))))
        df = df.dropna(subset=["pitch_type"])   # identical to m6_evaluate, so identical folds
        data[name], folds[name] = df, game_folds(df, K, SEED)
    union = sorted(set().union(*(d["pitch_type"].unique() for d in data.values())))
    rows, held = [], {}
    for f in range(K):
        test = {n: d[d["game_pk"].isin(folds[n][f])] for n, d in data.items()}
        train = {n: d[~d["game_pk"].isin(folds[n][f])] for n, d in data.items()}
        tf = TransformerSeq(union).fit(pd.concat(train.values()))
        print(f"fold {f}: transformer config (d, dropout, wd) = {tf.config}, {tf.epochs} epochs")
        for n, te in test.items():
            types = sorted(data[n]["pitch_type"].unique())
            y = te["pitch_type"].map({t: i for i, t in enumerate(types)}).to_numpy()
            row = {"pitcher": n, "fold": f, "test_pitches": len(te), "epochs": tf.epochs}
            preds = {"transformer": tf.predict(te, types)}
            for ref, feats in REFS.items():
                make = lambda t, h, k=feats: LogisticSeq(t, C=h, features=k)
                preds[ref] = make(types, tune(make, C_GRID, train[n], types)).fit(train[n]).predict(te)
            for m, p in preds.items():
                row[m] = log_loss(p, y)
                held.setdefault((n, m), ([], []))[0].append(p); held[(n, m)][1].append(y)
            rows.append(row)
    res = pd.DataFrame(rows)
    OUT.mkdir(exist_ok=True)
    res.to_csv(OUT / "m6_transformer_folds.csv", index=False)
    wavg = lambda g, c: np.average(g[c], weights=g.test_pitches)
    print(res.groupby("pitcher", sort=False).apply(lambda g: pd.Series({
        **{m: wavg(g, m) for m in [*REFS, "transformer"]},
        "gain_vs_tto": wavg(g, "logistic+tto") - wavg(g, "transformer"),
        "gain_vs_prev": wavg(g, "logistic+prev") - wavg(g, "transformer"),
        "folds_beat_tto": f"{(g.transformer < g['logistic+tto']).sum()}/{len(g)}",
        "folds_beat_prev": f"{(g.transformer < g['logistic+prev']).sum()}/{len(g)}"})).round(4).to_string())
    rel = pd.concat(reliability(np.vstack(ps), np.concatenate(ys)).assign(pitcher=n, model=m).reset_index()
                    for (n, m), (ps, ys) in held.items())
    rel.to_csv(OUT / "m6_transformer_reliability.csv", index=False)
    big = rel.query("n >= 100")
    print("\nlargest calibration gap in bins with n >= 100:")
    print((big.observed - big.mean_pred).abs().groupby([big.pitcher, big.model]).max().round(3).unstack().to_string())
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot([0, 1], [0, 1], "k:", lw=1)
    for n in PITCHERS:
        r = rel.query("pitcher == @n and model == 'transformer' and n >= 30")
        ax.plot(r.mean_pred, r.observed, "o-", label=n)
    ax.set(xlabel="predicted probability", ylabel="observed frequency",
           title=f"transformer ({K}-fold by game, bins n>=30)")
    ax.legend(); fig.tight_layout(); fig.savefig(OUT / "m6_transformer_calibration.png", dpi=120)


if __name__ == "__main__":
    sys.exit(main())
