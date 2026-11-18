"""M6 count baseline vs the no-context mix, held-out by game. Writes results/m6_baseline.csv + png."""
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
OUT = Path(__file__).resolve().parents[1] / "results"
SEED = 0


def main() -> None:
    rows, rel_all = [], {}
    for name, pid in PITCHERS.items():
        df = load_seasons(sorted(RAW_DIR.glob(f"{pid}_*.csv")))
        df = df.dropna(subset=["pitch_type", "balls", "strikes"])
        train, test = split_by_game(df, seed=SEED)
        types = sorted(df["pitch_type"].unique())
        idx = {t: i for i, t in enumerate(types)}
        y = test["pitch_type"].map(idx).to_numpy()
        model = CountBaseline(types).fit(train)
        p = model.predict(test)
        flat = np.tile(model.overall, (len(test), 1))   # reference: ignore the count entirely
        rows.append({"pitcher": name, "train_pitches": len(train), "test_pitches": len(test),
                     "test_games": test["game_pk"].nunique(),
                     "logloss_overall": log_loss(flat, y), "logloss_count": log_loss(p, y)})
        rel_all[name] = reliability(p, y)
    res = pd.DataFrame(rows).assign(gain=lambda d: d.logloss_overall - d.logloss_count)
    OUT.mkdir(exist_ok=True)
    res.to_csv(OUT / "m6_baseline.csv", index=False)
    print(res.round(4).to_string(index=False))
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot([0, 1], [0, 1], "k:", lw=1)
    for name, r in rel_all.items():
        ax.plot(r.mean_pred, r.observed, "o-", label=name)
    ax.set(xlabel="predicted probability", ylabel="observed frequency", title="Count baseline calibration")
    ax.legend(); fig.tight_layout(); fig.savefig(OUT / "m6_baseline_calibration.png", dpi=120)


if __name__ == "__main__":
    sys.exit(main())
