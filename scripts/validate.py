"""
Step 1 validation: how well does our physics reproduce real MLB pitches?

    python scripts/validate.py data/669373_2025-04-01_2025-09-30.csv
    python scripts/validate.py data/synthetic.csv --no-weather

What it does:
  0. Data check  - Statcast's own constant-acceleration fit should land on
                   plate_x/plate_z. If it doesn't, we're reading coordinates wrong.
  1. Convention  - Statcast's spin_axis should point the same way as the
                   observed (non-gravity) acceleration. Checks our sign convention.
  2. Baseline    - simulate every pitch from its state at y = 50 ft using
                   default Cd / spin efficiency; measure plate-location error.
  3. Calibrated  - fit Cd and spin efficiency per pitch type on 70% of pitches,
                   measure error on the held-out 30%.
  4. Bias        - the average signed miss per pitch type after calibration.
                   A consistent miss in one direction is movement that spin
                   alone doesn't explain (e.g. seam-shifted wake) - step 3 of
                   the roadmap.

Runs four variants (start state y50 or release, constant or per-game air
density). Outputs <out>/summary.csv and <out>/report.png (from release+game).
"""

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pitchsim.data import add_air_density, fetch_game_conditions
from pitchsim.physics import PhysicsParams, air_density, simulate
from pitchsim.statcast import (clean, fit_plate_crossing, fit_trajectory, release_state,
                               observed_break_direction_deg)
from pitchsim.validation import MIN_PITCHES, angle_diff, evaluate

RESULTS = Path(__file__).resolve().parents[1] / "results"

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3dd"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--out", help="output folder (default results/m1/<pitcher>)")
    ap.add_argument("--no-weather", action="store_true",
                    help="skip the per-game air density variants (synthetic data)")
    args = ap.parse_args()

    raw = pd.read_csv(args.csv)
    df = clean(raw)
    name = raw["player_name"].iloc[0] if "player_name" in raw else "pitcher"
    print(f"{name}: {len(df):,} usable pitches (of {len(raw):,})\n")

    # Fake pitches must never land next to real results.
    if "Synthetic" in name:
        out = RESULTS / "synthetic"
        if args.out and Path(args.out).resolve() != out.resolve():
            raise SystemExit(f"Synthetic data may only be written to {out}")
    else:
        slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
        out = Path(args.out) if args.out else RESULTS / "m1" / slug
    out.mkdir(parents=True, exist_ok=True)

    # 0. data check
    fpos, fvel, ftime = fit_plate_crossing(df)
    fit_err = np.hypot(fpos[:, 0] - df["plate_x"], fpos[:, 2] - df["plate_z"]) * 12
    print(f"[0] Statcast fit vs reported plate location: median {np.median(fit_err):.2f} in "
          f"(should be ~0)")

    # 1. convention check
    dang = angle_diff(observed_break_direction_deg(df), df["spin_axis"].to_numpy())
    print(f"[1] spin_axis vs observed break direction: median |diff| "
          f"{np.median(np.abs(dang)):.1f} deg (small = sign convention correct)\n")

    const = np.full(len(df), air_density(70.0, 0.0))
    variants = {"y50+const": ("y50", const), "release+const": ("release", const)}
    if not args.no_weather:
        game = add_air_density(df, fetch_game_conditions(df["game_pk"].unique()))["rho"].to_numpy()
        variants |= {"y50+game": ("y50", game), "release+game": ("release", game)}

    tables = []
    for label, (start, rho) in variants.items():
        summary, skipped = evaluate(df, start, rho)
        print(f"== {label} ==")
        for s in skipped:
            print(f"skipped {s}")
        if summary.empty:
            print("no pitch type could be evaluated\n")
            continue
        summary = summary.sort_values("n", ascending=False)
        show = summary[["pitch_name", "n", "mph", "rpm", "baseline_err_in", "cd", "spin_eff",
                        "calibrated_err_in", "p90_err_in", "bias_x_in", "bias_z_in",
                        "plate_speed_err_mph"]]
        print("Held-out (by game) plate-location error (inches), per pitch type:")
        print(show.round(2).to_string(index=False), "\n")
        tables.append(summary.assign(variant=label))
    if not tables:
        raise SystemExit(f"No pitch type has {MIN_PITCHES}+ pitches in held-out games; "
                         "use a longer date range.")

    all_rows = pd.concat(tables)
    cols = ["variant"] + [c for c in all_rows.columns if c != "variant"]
    all_rows[cols].round(3).to_csv(out / "summary.csv", index=False)
    print("bias_x/bias_z = average signed miss after calibration (x: + = catcher's right, "
          "z: + = too high).\nA large, consistent bias means movement the spin model "
          "can't explain.")

    # report from release+game (release+const when there is no weather data)
    report_label = "release+const" if args.no_weather else "release+game"
    summary = all_rows[all_rows["variant"] == report_label]
    if summary.empty:
        raise SystemExit(f"{report_label} has no results, so no report.png")
    make_report(df.assign(rho=variants[report_label][1]), summary, name, report_label, out)
    print(f"\nSaved {out / 'summary.csv'} and {out / 'report.png'}")


def style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.yaxis.label.set_color(MUTED)
    ax.xaxis.label.set_color(MUTED)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def make_report(df, summary, name, label, out):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [1, 1.3]})
    fig.patch.set_facecolor("white")

    # Panel 1: error per pitch type, baseline vs calibrated
    labels = summary["pitch_name"].tolist()
    x = np.arange(len(labels))
    w = 0.38
    b1 = ax1.bar(x - w/2 - 0.01, summary["baseline_err_in"], w, color=BLUE, label="Default physics")
    b2 = ax1.bar(x + w/2 + 0.01, summary["calibrated_err_in"], w, color=ORANGE, label="Calibrated")
    for bars in (b1, b2):
        ax1.bar_label(bars, fmt="%.1f", fontsize=8, color=INK, padding=2)
    ax1.set_xticks(x, labels, rotation=20, ha="right")
    ax1.set_ylabel("Median miss at plate (inches)")
    ax1.set_title("How far the simulated pitch lands from the real one",
                  loc="left", fontsize=11, color=INK)
    ax1.legend(frameon=False, fontsize=9)
    style(ax1)

    # Panel 2: side view of one real pitch vs our calibrated simulation
    top = summary.iloc[0]
    g = df[df["pitch_type"] == top["pitch_type"]]
    row = g.iloc[len(g) // 2]
    real = fit_trajectory(row)
    pos0, vel0 = release_state(row.to_frame().T.astype(
        {c: float for c in ["release_pos_x", "release_pos_y", "release_pos_z",
                            "vx0", "vy0", "vz0", "ax", "ay", "az"]}))
    sim = simulate(pos0[0], vel0[0], float(row["release_spin_rate"]), float(row["spin_axis"]),
                   PhysicsParams(cd=top["cd"], spin_efficiency=top["spin_eff"],
                                 rho=float(row["rho"])))
    ax2.plot(real[:, 1], real[:, 2], color=BLUE, linewidth=2, label="Statcast (real)")
    ax2.plot(sim["pos"][:, 1], sim["pos"][:, 2], color=ORANGE, linewidth=2,
             linestyle="--", label="Simulated (from release)")
    ax2.invert_xaxis()   # pitcher on the left, plate on the right
    ax2.set_xlabel("Distance from home plate (ft)")
    ax2.set_ylabel("Height (ft)")
    ax2.set_title(f"Side view: one {top['pitch_name']}", loc="left", fontsize=11, color=INK)
    ax2.legend(frameon=False, fontsize=9)
    style(ax2)

    fig.suptitle(f"{name}: physics model vs Statcast ({label})", x=0.01, ha="left",
                 fontsize=13, color=INK, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "report.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
