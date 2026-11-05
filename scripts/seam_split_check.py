"""
M3: does the pass count depend on which games were held out? REAL data only.

    python scripts/seam_split_check.py

Runs evaluate_pitcher on all five pitchers with game-split seeds 0-3 (seed 0 is the split every
other M3 result uses), default bound, same air. Writes results/m3/split_check.csv, one row per
split seed x pitcher x fitted pitch type, and prints graded / pass counts per split. Comparison
only: nothing else reads this file.
"""

import re
import unicodedata
from pathlib import Path

import pandas as pd

from pitchsim.data import add_air_density, fetch_game_conditions
from pitchsim.seam import evaluate_pitcher
from pitchsim.statcast import clean

ROOT = Path(__file__).resolve().parents[1]
SPLIT_SEEDS = range(4)
COLS = ["pitch_type", "n_test", "n_games_test", "err_m1_in", "err_per_pitch_in", "drop_in",
        "drop_lo_in", "drop_hi_in", "graded", "pass"]


def slug_of(name: str) -> str:
    plain = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", plain.lower()).strip("-")


def main():
    rows = []
    for path in sorted((ROOT / "data" / "raw").glob("*.csv")):
        if path.name == "game_conditions.csv":
            continue
        raw = pd.read_csv(path)
        slug = slug_of(raw["player_name"].iloc[0])
        df = clean(raw)
        rho = add_air_density(df, fetch_game_conditions(df["game_pk"].unique()))["rho"].to_numpy()
        for s in SPLIT_SEEDS:
            summary = evaluate_pitcher(df, rho, split_seed=s)[0].round(3)
            rows.append(summary[COLS].assign(pitcher=slug, split_seed=s))
            print(slug, "split seed", s, "done", flush=True)
    table = pd.concat(rows, ignore_index=True)
    table["graded"] = table["graded"].astype(bool)
    table["pass"] = table["pass"].astype(str) == "True"
    table.to_csv(ROOT / "results" / "m3" / "split_check.csv", index=False)

    g = table[table["graded"]]
    print(g.groupby("split_seed").agg(graded=("graded", "size"), passed=("pass", "sum"),
                                      at_1in=("err_per_pitch_in", lambda x: (x <= 1).sum())))
    fails = g[~g["pass"]].groupby(["pitcher", "pitch_type"])["split_seed"].agg(list)
    print("failing rows (split seeds):\n", fails.to_string())


if __name__ == "__main__":
    main()
