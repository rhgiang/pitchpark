"""M4 checks 2 and 3: perceived velocity vs Statcast's effective_speed, and our approach angles.

    python scripts/m4_checks.py data/raw/<pitcher>.csv [more.csv ...]

Held-out games only (validation.split_by_game, seed 0, the M1-M3 split). Writes
results/m4/checks.csv with one row per pitcher and pitch type plus an "ALL" row per pitcher:

- perc_med_abs_diff_mph: median |perceived_velocity_mph - effective_speed| on held-out pitches,
  with tunneling.PERCEIVED_REF_EXT_FT as it is now; perc_med_diff_mph is the signed median.
- perc_med_abs_diff_mph_ref6: the same with the Task 2 assumption (6.0 ft), for before/after.
- vaa_med_deg / haa_med_deg: median approach angles from the Statcast-fit velocity at the
  front of the plate (what the viewer shows).

It also prints the reference extension that best fits effective_speed on the TRAINING games of
all the given pitchers pooled (grid search, 0.01 ft steps, minimising the median absolute
difference). That is the only thing PERCEIVED_REF_EXT_FT was tuned on (see tunneling.py).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from pitchsim.statcast import clean, fit_plate_crossing
from pitchsim.tunneling import (PERCEIVED_REF_EXT_FT, RUBBER_Y_FT, approach_angles_deg,
                                perceived_velocity_mph)
from pitchsim.validation import split_by_game

ROOT = Path(__file__).resolve().parents[1]


def perceived_at_ref(speed_mph, extension_ft, ref_ext_ft: float) -> np.ndarray:
    """perceived_velocity_mph's formula with a different reference extension (ft)."""
    return np.asarray(speed_mph) * (RUBBER_Y_FT - ref_ext_ft) / (RUBBER_Y_FT - np.asarray(extension_ft))


def best_ref_ext_ft(df: pd.DataFrame) -> tuple[float, float]:
    """Reference extension (ft) minimising median |perceived - effective_speed|, and that median."""
    grid = np.round(np.arange(5.5, 7.0001, 0.01), 2)
    err = [np.median(np.abs(perceived_at_ref(df["release_speed"], df["release_extension"], r)
                            - df["effective_speed"])) for r in grid]
    k = int(np.argmin(err))
    return float(grid[k]), float(err[k])


def _row(pitcher: str, ptype: str, d: pd.DataFrame) -> dict:
    s = d.dropna(subset=["effective_speed", "release_extension"])
    diff = perceived_velocity_mph(s["release_speed"], s["release_extension"]) - s["effective_speed"]
    diff6 = perceived_at_ref(s["release_speed"], s["release_extension"], 6.0) - s["effective_speed"]
    return {"pitcher": pitcher, "pitch_type": ptype, "n_heldout": len(d), "n_speed": len(s),
            "perc_med_abs_diff_mph": round(float(np.median(np.abs(diff))), 3),
            "perc_med_diff_mph": round(float(np.median(diff)), 3),
            "perc_med_abs_diff_mph_ref6": round(float(np.median(np.abs(diff6))), 3),
            "vaa_med_deg": round(float(d["vaa"].median()), 2),
            "haa_med_deg": round(float(d["haa"].median()), 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="+")
    ap.add_argument("--out", default=str(ROOT / "results" / "m4" / "checks.csv"))
    args = ap.parse_args()

    rows, train_all = [], []
    for csv in args.csv:
        df = clean(pd.read_csv(csv))
        train, test = split_by_game(df, seed=0)
        train_all.append(train.dropna(subset=["effective_speed", "release_extension"]))
        _, vel, _ = fit_plate_crossing(test)
        vaa, haa = approach_angles_deg(vel)
        test = test.copy().assign(vaa=vaa, haa=haa)  # copy() defragments (pandas warns otherwise)
        name = str(df["player_name"].iloc[0])
        rows += [_row(name, k, g) for k, g in test.groupby("pitch_type")]
        rows.append(_row(name, "ALL", test))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    table.to_csv(out, index=False)
    with pd.option_context("display.width", 200):
        print(table.to_string(index=False))
    ref, err = best_ref_ext_ft(pd.concat(train_all))
    print(f"\nPERCEIVED_REF_EXT_FT = {PERCEIVED_REF_EXT_FT} ft in use. Best fit on training games "
          f"(pooled, {sum(map(len, train_all))} pitches): {ref:.2f} ft, median |diff| {err:.3f} mph")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
