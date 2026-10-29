"""
Step 1 validation: how well does our physics reproduce real MLB pitches?

    python scripts/validate.py data/669373_2025-04-01_2025-09-30.csv

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

Outputs results/summary.csv and results/report.png.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from pitchsim import physics
from pitchsim.physics import PhysicsParams, air_density, simulate, simulate_batch
from pitchsim.statcast import (clean, fit_plate_crossing, fit_trajectory, initial_state,
                      observed_break_direction_deg)

FT_PER_S_TO_MPH = 0.681818
MIN_PITCHES = 30
MAX_TRAIN = 400
RESULTS = Path(__file__).resolve().parents[1] / "results"

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3dd"


def run_sim(df, cd, eff, rho):
    pos0, vel0 = initial_state(df)
    return simulate_batch(pos0, vel0, df["release_spin_rate"].to_numpy(),
                          df["spin_axis"].to_numpy(), cd, eff, rho=rho)


def residuals(df, fit_vel, cd, eff, rho):
    pos, vel, _ = run_sim(df, cd, eff, rho)
    dx = (pos[:, 0] - df["plate_x"].to_numpy()) * 12
    dz = (pos[:, 2] - df["plate_z"].to_numpy()) * 12
    dv = (np.linalg.norm(vel, axis=1) - np.linalg.norm(fit_vel, axis=1)) * FT_PER_S_TO_MPH
    return dx, dz, dv


def calibrate(train, rho):
    _, fit_vel, _ = fit_plate_crossing(train)

    def resid(params):
        dx, dz, dv = residuals(train, fit_vel, params[0], params[1], rho)
        return np.concatenate([dx, dz, dv])   # inches and mph, roughly comparable

    res = least_squares(resid, x0=[0.33, 0.9], bounds=([0.15, 0.2], [0.6, 1.0]),
                        diff_step=1e-3)
    return res.x


def angle_diff(a, b):
    return (a - b + 180) % 360 - 180


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--temp", type=float, default=70.0, help="air temperature, F")
    ap.add_argument("--elevation", type=float, default=0.0, help="feet above sea level")
    args = ap.parse_args()

    raw = pd.read_csv(args.csv)
    df = clean(raw)
    rho = air_density(args.temp, args.elevation)
    name = raw["player_name"].iloc[0] if "player_name" in raw else "pitcher"
    print(f"{name}: {len(df):,} usable pitches (of {len(raw):,}); air density {rho:.4f} lb/ft^3\n")

    # 0. data check
    fpos, fvel, ftime = fit_plate_crossing(df)
    fit_err = np.hypot(fpos[:, 0] - df["plate_x"], fpos[:, 2] - df["plate_z"]) * 12
    print(f"[0] Statcast fit vs reported plate location: median {np.median(fit_err):.2f} in "
          f"(should be ~0)")

    # 1. convention check
    dang = angle_diff(observed_break_direction_deg(df), df["spin_axis"].to_numpy())
    print(f"[1] spin_axis vs observed break direction: median |diff| "
          f"{np.median(np.abs(dang)):.1f} deg (small = sign convention correct)\n")

    rng = np.random.default_rng(0)
    rows = []
    for ptype, g in df.groupby("pitch_type"):
        if len(g) < MIN_PITCHES:
            continue
        g = g.reset_index(drop=True)
        idx = rng.permutation(len(g))
        cut = int(0.7 * len(g))
        train, test = g.iloc[idx[:cut]], g.iloc[idx[cut:]]
        if len(train) > MAX_TRAIN:
            train = train.sample(MAX_TRAIN, random_state=0)
        _, test_fit_vel, _ = fit_plate_crossing(test)

        d = PhysicsParams()
        bx, bz, bv = residuals(test, test_fit_vel, d.cd, d.spin_efficiency, rho)
        cd, eff = calibrate(train, rho)
        cx, cz, cv = residuals(test, test_fit_vel, cd, eff, rho)

        rows.append({
            "pitch_type": ptype,
            "pitch_name": g["pitch_name"].iloc[0] if "pitch_name" in g else ptype,
            "n": len(g),
            "mph": g["release_speed"].mean(),
            "rpm": g["release_spin_rate"].mean(),
            "baseline_err_in": np.median(np.hypot(bx, bz)),
            "cd": cd,
            "spin_eff": eff,
            "calibrated_err_in": np.median(np.hypot(cx, cz)),
            "p90_err_in": np.percentile(np.hypot(cx, cz), 90),
            "bias_x_in": cx.mean(),
            "bias_z_in": cz.mean(),
            "plate_speed_err_mph": np.median(np.abs(cv)),
        })

    if not rows:
        raise SystemExit(f"No pitch type has {MIN_PITCHES}+ pitches; use a longer date range.")
    summary = pd.DataFrame(rows).sort_values("n", ascending=False)
    RESULTS.mkdir(exist_ok=True)
    summary.round(3).to_csv(RESULTS / "summary.csv", index=False)

    show = summary[["pitch_name", "n", "mph", "rpm", "baseline_err_in", "cd", "spin_eff",
                    "calibrated_err_in", "p90_err_in", "bias_x_in", "bias_z_in",
                    "plate_speed_err_mph"]].copy()
    print("Held-out plate-location error (inches), per pitch type:")
    print(show.round(2).to_string(index=False))
    print("\nbias_x/bias_z = average signed miss after calibration (x: + = catcher's right, "
          "z: + = too high).\nA large, consistent bias means movement the spin model "
          "can't explain.")

    make_report(df, summary, rho, name)
    print(f"\nSaved {RESULTS / 'summary.csv'} and {RESULTS / 'report.png'}")


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


def make_report(df, summary, rho, name):
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
    pos0, vel0 = initial_state(row.to_frame().T.astype(
        {c: float for c in ["release_pos_x", "release_pos_y", "release_pos_z",
                            "vx0", "vy0", "vz0", "ax", "ay", "az"]}))
    sim = simulate(pos0[0], vel0[0], float(row["release_spin_rate"]), float(row["spin_axis"]),
                   PhysicsParams(cd=top["cd"], spin_efficiency=top["spin_eff"], rho=rho))
    ax2.plot(real[:, 1], real[:, 2], color=BLUE, linewidth=2, label="Statcast (real)")
    ax2.plot(sim["pos"][:, 1], sim["pos"][:, 2], color=ORANGE, linewidth=2,
             linestyle="--", label="Simulated (from y = 50 ft)")
    ax2.invert_xaxis()   # pitcher on the left, plate on the right
    ax2.set_xlabel("Distance from home plate (ft)")
    ax2.set_ylabel("Height (ft)")
    ax2.set_title(f"Side view: one {top['pitch_name']}", loc="left", fontsize=11, color=INK)
    ax2.legend(frameon=False, fontsize=9)
    style(ax2)

    fig.suptitle(f"{name}: physics model vs Statcast", x=0.01, ha="left",
                 fontsize=13, color=INK, fontweight="bold")
    fig.tight_layout()
    fig.savefig(RESULTS / "report.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
