"""
M2: fit and score the per-pitcher generative model.

    python scripts/train_generative.py data/raw/<pitcher>.csv
    python scripts/train_generative.py data/synthetic.csv --no-weather

Writes <out>/summary.csv, ks.csv, context.csv, physical.csv, skipped.txt and one <pitch_type>.png per pitch type
(real held-out vs generated). Default <out> is results/m2/<pitcher>.
"""

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from pitchsim.data import add_air_density, fetch_game_conditions, load_pitcher
from pitchsim.generative import circular_mean_deg, evaluate_pitcher, pitch_count_in_game, unwrap_deg
from pitchsim.physics import air_density
from pitchsim.statcast import clean

RESULTS = Path(__file__).resolve().parents[1] / "results"

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3dd"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="+", help="one pitcher's CSVs (one per season)")
    ap.add_argument("--cross-season", action="store_true",
                    help="train on earlier seasons, test on the newest (out: <pitcher>-cross-season)")
    ap.add_argument("--out", help="output folder (default results/m2/<pitcher>)")
    ap.add_argument("--no-weather", action="store_true",
                    help="use a constant air density (synthetic data)")
    args = ap.parse_args()

    raw = load_pitcher(args.csv, args.cross_season)
    # Count on the full CSV (all pitch types) before clean() drops rows.
    if {"game_pk", "at_bat_number", "pitch_number"} <= set(raw):
        raw["pitch_count"] = pitch_count_in_game(raw)
    df = clean(raw)
    name = raw["player_name"].iloc[0] if "player_name" in raw else "pitcher"
    print(f"{name}: {len(df):,} usable pitches (of {len(raw):,})\n")

    # Fake pitches must never land next to real results.
    if "Synthetic" in name:
        out = RESULTS / "synthetic"
        if args.out and Path(args.out).resolve() != out.resolve():
            raise SystemExit(f"Synthetic data may only be written to {out}")
    else:
        slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") + ("-cross-season" if args.cross_season else "")
        out = Path(args.out) if args.out else RESULTS / "m2" / slug
    out.mkdir(parents=True, exist_ok=True)

    if args.no_weather:
        rho = pd.Series(air_density(70.0, 0.0), index=df.index)
    else:
        rho = add_air_density(df, fetch_game_conditions(df["game_pk"].unique()))["rho"]

    summary, ks, skipped, clouds, context, physical = evaluate_pitcher(df, rho.to_numpy())

    (out / "skipped.txt").write_text("".join(s + "\n" for s in skipped))
    for s in skipped:
        print(f"skipped {s}")
    summary.round(3).to_csv(out / "summary.csv", index=False)
    ks.round(4).to_csv(out / "ks.csv", index=False)
    context.round(4).to_csv(out / "context.csv", index=False)
    physical.round(4).to_csv(out / "physical.csv", index=False)

    if summary.empty:
        print("no pitch type could be evaluated")
        return
    print(summary.round(3).to_string(index=False), "\n")
    for ptype, c in clouds.items():
        make_plot(name, ptype, summary[summary["pitch_type"] == ptype], c, out)
    print(f"Saved summary.csv, ks.csv, context.csv, physical.csv, skipped.txt and {len(clouds)} plot(s) to {out}")


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


def make_plot(name, ptype, summ, c, out):
    """Five panels: real held-out (blue) vs generated (orange), plus movement clouds."""
    real, gen = c["real"], c["generated"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.patch.set_facecolor("white")
    ax_rel, ax_vs, ax_axis, ax_ext, ax_mov, ax_off = axes.ravel()
    ax_off.axis("off")

    for ax, x, y, xl, yl, t in [
            (ax_rel, "release_pos_x", "release_pos_z", "release side (ft)", "release height (ft)",
             "release point"),
            (ax_vs, "release_speed", "release_spin_rate", "velocity (mph)", "spin rate (rpm)",
             "velocity vs spin")]:
        ax.scatter(gen[x], gen[y], s=6, color=ORANGE, alpha=0.3, label="generated")
        ax.scatter(real[x], real[y], s=10, color=BLUE, alpha=0.7, label="real held-out")
        ax.set_xlabel(xl); ax.set_ylabel(yl)
        ax.set_title(t, loc="left", fontsize=11, color=INK)
        style(ax)
    for ax, col, xl in [(ax_axis, "spin_axis", "spin axis (deg)"),
                        (ax_ext, "release_extension", "extension (ft)")]:
        r, g = real[col].to_numpy(), gen[col].to_numpy()
        if col == "spin_axis":
            # Circular: unwrap both samples about one shared center so a 350-20 deg
            # pitch is one bump, not two spikes at opposite ends.
            ctr = circular_mean_deg(pd.concat([real[col], gen[col]]))
            r, g = unwrap_deg(r, ctr), unwrap_deg(g, ctr)
            xl = f"spin axis (deg, centered on {ctr:.0f}; mod 360)"
        bins = pd.cut(pd.Series([*r, *g]), 25, retbins=True)[1]
        ax.hist(g, bins=bins, density=True, color=ORANGE, alpha=0.5, label="generated")
        ax.hist(r, bins=bins, density=True, color=BLUE, alpha=0.5, label="real held-out")
        ax.set_xlabel(xl); ax.set_ylabel("density")
        ax.set_title(xl.split(" (")[0], loc="left", fontsize=11, color=INK)
        style(ax)
    for key, col, lab, s in [("move_statcast", BLUE, "real (Statcast)", 10),
                             ("move_real_m1", MUTED, "real flown by M1", 10),
                             ("move_gen_m1", ORANGE, "generated flown by M1", 6)]:
        hb, vb = c[key]
        ax_mov.scatter(hb, vb, s=s, color=col, alpha=0.4, label=lab)
    ax_mov.set_xlabel("horizontal break (in)"); ax_mov.set_ylabel("vertical break (in)")
    ax_mov.set_title("movement — not graded until M3", loc="left", fontsize=11, color=INK)
    style(ax_mov)
    for ax in (ax_rel, ax_axis, ax_mov):
        leg = ax.legend(fontsize=8, frameon=True, framealpha=1, facecolor="white", edgecolor=GRID)  # opaque: data points showed through the markers
        for h in leg.legend_handles:   # scatter alpha is baked into the handle; match the cloud colour
            h.set_alpha(1)

    bits = []
    for _, r in summ.iterrows():
        status = "ungraded" if not r["graded"] else ("pass" if r["pass"] else "fail")
        bits.append(f"{r['model']} AUC {r['auc']:.2f} ({status})")
    fig.suptitle(f"{name}: {ptype}, " + ", ".join(bits), x=0.01, ha="left",
                 fontsize=13, color=INK, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / f"{ptype}.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
