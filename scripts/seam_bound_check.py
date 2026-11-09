"""
M3: the record of moving M1's spin-efficiency lower bound from 0.20 to 0.01. REAL data only.

    python scripts/seam_bound_check.py

Run train_seam.py first: new_* columns are the default results in results/m3/<pitcher>/. This
reruns each pitcher with the old bound for old_* columns, same air and held-out games, and writes
results/m3/bound_check.csv, one row per fitted pitch type. on_old_bound marks rows the old fit
pinned at 0.20. It also writes results/m3/spearman.csv: Spearman rank correlation of |axis gap|
with push size over the graded rows, at each bound. Comparison only: nothing else reads these.
"""

import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from pitchsim.data import add_air_density, fetch_game_conditions
from pitchsim.seam import evaluate_pitcher
from pitchsim.statcast import clean

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "m3"
OLD_BOUND = 0.20
COLS = ["spin_eff", "cd", "err_m1_in", "err_fixed_in", "err_per_pitch_in", "drop_in",
        "drop_lo_in", "drop_hi_in", "pp_vs_fixed_in", "pp_vs_fixed_lo_in", "pp_vs_fixed_hi_in",
        "push_x_fts2", "push_z_fts2", "axis_gap_deg", "graded", "pass"]


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
        new = pd.read_csv(RESULTS / slug / "summary.csv")
        df = clean(raw)
        rho = add_air_density(df, fetch_game_conditions(df["game_pk"].unique()))["rho"]
        old = evaluate_pitcher(df, rho.to_numpy(), eff_min=OLD_BOUND)[0].round(3)
        old = old.set_index("pitch_type").loc[new["pitch_type"]].reset_index()
        out = new[["pitch_type"]].assign(pitcher=slug,
                                         on_old_bound=old["spin_eff"] <= OLD_BOUND + 5e-4)
        rows.append(pd.concat([out, old[COLS].add_prefix("old_"), new[COLS].add_prefix("new_")],
                              axis=1))
    table = pd.concat(rows, ignore_index=True)
    table.to_csv(RESULTS / "bound_check.csv", index=False)

    g = table[table["new_graded"].astype(bool)]
    for side in ("old", "new"):
        print(f"{side}: {(g[side + '_pass'].astype(str) == 'True').sum()} of {len(g)} pass, "
              f"{(g[side + '_pp_vs_fixed_lo_in'] > 0).sum()} beat fixed (lower end above zero)")
    sp = []
    for side in ("old", "new"):
        size = np.hypot(g[side + "_push_x_fts2"], g[side + "_push_z_fts2"])
        r = spearmanr(g[side + "_axis_gap_deg"].abs(), size)
        sp.append({"bound": side, "n_graded": len(g), "spearman": r.statistic, "p_value": r.pvalue})
    sp = pd.DataFrame(sp)
    sp.to_csv(RESULTS / "spearman.csv", index=False)
    print(sp.round(3).to_string(index=False))
    off = table[~table["on_old_bound"]]
    print("largest per-pitch median shift off the old bound (in):",
          (off["old_err_per_pitch_in"] - off["new_err_per_pitch_in"]).abs().max())
    print(table[table["on_old_bound"]].T.to_string())
    print(f"Saved {RESULTS / 'bound_check.csv'}")


if __name__ == "__main__":
    main()
