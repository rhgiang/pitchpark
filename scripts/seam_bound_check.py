"""
M3: what does M1's 0.20 spin-efficiency bound do to the results? REAL data only.

    python scripts/seam_bound_check.py

Run train_seam.py first. For each pitcher in results/m3/ with a pitch type on the bound, reruns
evaluate_pitcher with the bound lowered to EFF_FLOOR and takes the on-bound rows from that run.
Every other row keeps its reported values. Writes results/m3/bound_check.csv with one row per
fitted pitch type (reported_* and lowered_* columns) so the counts can be read from the file.
Nothing else changes: the default bound stays 0.20.
"""

import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

from pitchsim.data import add_air_density, fetch_game_conditions
from pitchsim.seam import evaluate_pitcher
from pitchsim.statcast import clean

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "m3"
EFF_FLOOR = 0.01   # Stays positive (efficiency is a share of spin) but low enough not to bind.
ON_BOUND = 0.2005  # summary.csv rounds to 3 decimals
COLS = ["spin_eff", "cd", "err_m1_in", "err_fixed_in", "err_per_pitch_in", "drop_in", "drop_lo_in",
        "drop_hi_in", "pp_vs_fixed_in", "pp_vs_fixed_lo_in", "pp_vs_fixed_hi_in", "graded", "pass"]


def slug_of(name: str) -> str:
    plain = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", plain.lower()).strip("-")


def main():
    rows, drift = [], []
    for path in sorted((ROOT / "data" / "raw").glob("*.csv")):
        if path.name == "game_conditions.csv":
            continue
        raw = pd.read_csv(path)
        slug = slug_of(raw["player_name"].iloc[0])
        reported = pd.read_csv(RESULTS / slug / "summary.csv").astype({"pass": object})
        on_bound = reported["spin_eff"] <= ON_BOUND
        lowered = reported.copy()
        if on_bound.any():
            df = clean(raw)
            rho = add_air_density(df, fetch_game_conditions(df["game_pk"].unique()))["rho"]
            rerun = evaluate_pitcher(df, rho.to_numpy(), eff_min=EFF_FLOOR)[0].round(3)
            rerun = rerun.set_index("pitch_type").loc[reported["pitch_type"]].reset_index()
            lowered.loc[on_bound, COLS] = rerun.loc[on_bound, COLS].to_numpy()
            # The other rows didn't bind, so a lower bound shouldn't move them. Check that.
            shift = (rerun["err_per_pitch_in"] - reported["err_per_pitch_in"]).abs()
            drift.append(shift[~on_bound].max())
        out = reported[["pitch_type"]].assign(pitcher=slug, on_bound=on_bound, eff_floor=EFF_FLOOR)
        rows.append(pd.concat([out, reported[COLS].add_prefix("reported_"),
                               lowered[COLS].add_prefix("lowered_")], axis=1))
    table = pd.concat(rows, ignore_index=True)
    table.to_csv(RESULTS / "bound_check.csv", index=False)

    g = table[table["reported_graded"].astype(bool)]
    for side in ("reported", "lowered"):
        print(f"{side}: {(g[side + '_pass'].astype(str) == 'True').sum()} of {len(g)} pass, "
              f"{(g[side + '_pp_vs_fixed_lo_in'] > 0).sum()} beat fixed (lower end above zero)")
    print("largest per-pitch median shift on rows off the bound (in):", max(drift))
    print(table[table["on_bound"]].drop(columns="eff_floor").T.to_string())
    print(f"Saved {RESULTS / 'bound_check.csv'}")


if __name__ == "__main__":
    main()
