"""
M3: fit and score the extra push (the part of a pitch's movement M1's spin can't explain).

    python scripts/train_seam.py data/raw/<pitcher>.csv
    python scripts/train_seam.py data/synthetic.csv --no-weather

Writes <out>/summary.csv, knock_on.csv, skipped.txt and one <pitch_type>.png per pitch type
(held-out misses, M1 only vs M1 + per-pitch push). Default <out> is results/m3/<pitcher>.
"""

import argparse
import re
import unicodedata
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from pitchsim.data import add_air_density, fetch_game_conditions, load_pitcher
from pitchsim.physics import air_density
from pitchsim.seam import evaluate_pitcher
from pitchsim.statcast import clean

RESULTS = Path(__file__).resolve().parents[1] / "results"

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3dd"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="+", help="one pitcher's CSVs (one per season)")
    ap.add_argument("--cross-season", action="store_true",
                    help="train on earlier seasons, test on the newest (out: <pitcher>-cross-season)")
    ap.add_argument("--out", help="output folder (default results/m3/<pitcher>)")
    ap.add_argument("--no-weather", action="store_true",
                    help="use a constant air density (synthetic data)")
    args = ap.parse_args()

    raw = load_pitcher(args.csv, args.cross_season)
    df = clean(raw)
    name = raw["player_name"].iloc[0] if "player_name" in raw else "pitcher"
    # Strip accents: the slug regex would turn an accented letter into a hyphen, and a Windows
    # console can't always print one.
    plain = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    print(f"{plain}: {len(df):,} usable pitches (of {len(raw):,})\n")

    # Fake pitches must never land next to real results.
    if "Synthetic" in name:
        out = RESULTS / "synthetic"
        if args.out and Path(args.out).resolve() != out.resolve():
            raise SystemExit(f"Synthetic data may only be written to {out}")
    else:
        slug = re.sub(r"[^a-z0-9]+", "-", plain.lower()).strip("-") + ("-cross-season" if args.cross_season else "")
        out = Path(args.out) if args.out else RESULTS / "m3" / slug
    out.mkdir(parents=True, exist_ok=True)

    if args.no_weather:
        rho = pd.Series(air_density(70.0, 0.0), index=df.index)
    else:
        rho = add_air_density(df, fetch_game_conditions(df["game_pk"].unique()))["rho"]

    summary, knock, skipped, clouds = evaluate_pitcher(df, rho.to_numpy())

    (out / "skipped.txt").write_text("".join(s + "\n" for s in skipped))
    for s in skipped:
        print(f"skipped {s}")
    summary.round(3).to_csv(out / "summary.csv", index=False)
    knock.round(4).to_csv(out / "knock_on.csv", index=False)

    if summary.empty:
        print("no pitch type could be evaluated")
        return
    print(summary.drop(columns="pitch_name").round(3).to_string(index=False), "\n")
    for ptype, c in clouds.items():
        make_plot(name, summary[summary["pitch_type"] == ptype].iloc[0], c, out)
    print(f"Saved summary.csv, knock_on.csv, skipped.txt and {len(clouds)} plot(s) to {out}")


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


def make_plot(name, r, c, out):
    """Left: held-out misses in the plate plane. Right: the three medians with the drop interval."""
    fig, (ax_miss, ax_bar) = plt.subplots(1, 2, figsize=(12, 5.5))
    fig.patch.set_facecolor("white")

    ax_miss.axhline(0, color=GRID, linewidth=1)
    ax_miss.axvline(0, color=GRID, linewidth=1)
    ax_miss.scatter(c["dx_m1_in"], c["dz_m1_in"], s=14, color=BLUE, alpha=0.6, label="M1 only")
    ax_miss.scatter(c["dx_pp_in"], c["dz_pp_in"], s=14, color=ORANGE, alpha=0.6,
                    label="M1 + per-pitch push")
    ax_miss.set_aspect("equal", adjustable="datalim")
    ax_miss.set_xlabel("miss x, simulated minus real (in)")
    ax_miss.set_ylabel("miss z, simulated minus real (in)")
    ax_miss.set_title("held-out plate misses", loc="left", fontsize=11, color=INK)
    style(ax_miss)
    leg = ax_miss.legend(fontsize=8, frameon=True, framealpha=1, facecolor="white", edgecolor=GRID)
    for h in leg.legend_handles:   # scatter alpha is baked into the handle; match the cloud colour
        h.set_alpha(1)

    # The interval is on the drop, so draw it where the drop lands: M1's median minus the drop.
    medians = [r["err_m1_in"], r["err_fixed_in"], r["err_per_pitch_in"]]
    ax_bar.bar(["M1 only", "M1 + fixed", "M1 + per-pitch"], medians, width=0.55,
               color=[BLUE, MUTED, ORANGE])
    lo, hi = r["err_m1_in"] - r["drop_hi_in"], r["err_m1_in"] - r["drop_lo_in"]
    ax_bar.errorbar([2], [medians[2]], yerr=[[max(medians[2] - lo, 0)], [max(hi - medians[2], 0)]],
                    color=INK, capsize=4, linewidth=1.2)
    for i, m in enumerate(medians):   # above the bar, and above the error bar on per-pitch
        ax_bar.text(i, max(m, hi) if i == 2 else m, f"{m:.2f}", ha="center", va="bottom",
                    fontsize=9, color=INK)
    ax_bar.set_ylabel("median miss (in)")
    ax_bar.set_title(f"drop {r['drop_in']:.2f} in, 95% interval "
                     f"[{r['drop_lo_in']:.2f}, {r['drop_hi_in']:.2f}] (bar on per-pitch)",
                     loc="left", fontsize=11, color=INK)
    style(ax_bar)

    status = "ungraded" if not r["graded"] else ("pass" if r["pass"] else "fail")
    fig.suptitle(f"{name}: {r['pitch_type']}, held-out games ({r['n_test']} pitches, "
                 f"{r['n_games_test']} games), {status}", x=0.01, ha="left",
                 fontsize=13, color=INK, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / f"{r['pitch_type']}.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
