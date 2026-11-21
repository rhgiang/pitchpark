"""M7 check, real side: the pitcher's held-out at-bats as summary statistics with 95% intervals.

    python scripts/m7_real_stats.py --all          # -> results/m7/<slug>-real.json (local)
    python scripts/m7_real_stats.py sale-chris

Data = all seasons pooled, split_by_game(seed=0) TEST games (the same split m7_export.py trains the
M6 models on). Per stand: n_ab, mix_by_count, zone_rate, first_pitch_strike, pitches_per_ab,
outcomes. Intervals: 2,000 game-level bootstrap resamples (seed 0): games are the unit because
pitches within a game share umpire, park and form. See results/M7.md for definitions.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from m6_location import add_location
from m7_export import FACE_DIR, SLUGS
from pitchsim.data import RAW_DIR, load_seasons
from pitchsim.sequence.context import add_context
from pitchsim.sequence.hitter import outcome_of, zone_bin
from pitchsim.validation import split_by_game

OUT = Path(__file__).resolve().parents[1] / "results" / "m7"
COUNTS = [f"{b}-{s}" for b in range(4) for s in range(3)]
N_BOOT = 2000
FPS_DESC = {"called_strike", "automatic_strike"}   # a take that is a strike (sim: isStrike on the take)


def _ab_table(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """One row per kept at-bat (terminal class + stand); exclusions are counted, not scored."""
    ex = {"hbp": 0, "truncated": 0, "intentional_walk": 0, "missing_data": 0}
    keep = []
    for (g, n), a in df.sort_values("pitch_number").groupby(["game_pk", "at_bat_number"], sort=False):
        last = a.iloc[-1]
        ev = last["events"] if isinstance(last["events"], str) else ""
        if last["description"] == "hit_by_pitch" or ev == "hit_by_pitch":
            ex["hbp"] += 1
        elif ev == "intent_walk":
            ex["intentional_walk"] += 1
        elif a["x_ft"].isna().any() or a["pitch_type"].isna().any() or a["description"].map(outcome_of).isna().any():
            ex["missing_data"] += 1
        elif ev.startswith("strikeout"):
            keep.append((g, n, a["stand"].iloc[0], "strikeout"))
        elif ev == "walk":
            keep.append((g, n, a["stand"].iloc[0], "walk"))
        elif last["description"] == "hit_into_play" and not ev.startswith(("caught_stealing", "pickoff", "stolen")):
            keep.append((g, n, a["stand"].iloc[0], "in_play"))
        else:                       # inning ended on a non-pitch event (caught stealing, truncated_pa ...)
            ex["truncated"] += 1
    return pd.DataFrame(keep, columns=["game_pk", "at_bat_number", "stand_ab", "end"]), ex


def _ci(num, den, est):
    """Bootstrap ratio from (n_boot,) sums. The estimate is clamped inside its own interval
    (a percentile interval of a tiny sample can miss the point estimate)."""
    lo, hi = np.percentile(num / np.maximum(den, 1), [2.5, 97.5])
    return {"est": float(est), "lo": float(min(lo, est)), "hi": float(max(hi, est))}


def real_stats(df: pd.DataFrame, bank_types: list[str], zone: dict, n_boot: int = N_BOOT, seed: int = 0) -> dict:
    """df: held-out pitches with context columns (count, x_ft, z_ft, stand, events, description...)."""
    abs_, ex = _ab_table(df)
    d = df.merge(abs_, on=["game_pk", "at_bat_number"])
    d["in_zone"] = False
    for s, z in zone.items():       # same zone rule as the engine's isStrike (heart / edge_in bins)
        m = d["stand_ab"] == s
        if m.any():
            d.loc[m, "in_zone"] = zone_bin(d.loc[m, "x_ft"], d.loc[m, "z_ft"], z["top_ft"], z["bot_ft"]).isin(["heart", "edge_in"])
    oc = d["description"].map(outcome_of)
    d["fps"] = (d["count"] == "0-0") & (oc.isin(["whiff", "foul", "in_play"]) | d["description"].isin(FPS_DESC))
    games = np.array(sorted(df["game_pk"].unique()))
    G = len(games)
    gi = {g: i for i, g in enumerate(games)}
    idx = np.random.default_rng(seed).integers(0, G, (n_boot, G))
    ti = {t: i for i, t in enumerate(bank_types)}

    def per(frame, w=None):         # per-game sum (or count) -> (G,)
        return np.bincount(frame["game_pk"].map(gi), weights=w, minlength=G).astype(float)

    tot = per(abs_)
    out = {"excluded_abs": ex, "n_ab_total": len(abs_), "bank_types": bank_types, "n_games": G,
           "excluded_type_share": float(1 - d["pitch_type"].isin(bank_types).mean()), "stand_share": {}}
    for s in ("L", "R"):
        c = per(abs_[abs_["stand_ab"] == s])
        out["stand_share"][s] = _ci(c[idx].sum(1), tot[idx].sum(1), c.sum() / tot.sum())
        a, p = abs_[abs_["stand_ab"] == s], d[d["stand_ab"] == s]
        if a.empty:
            continue
        n_ab, n_p = per(a), per(p)
        zone_n, fps_n = per(p, p["in_zone"].astype(float)), per(p, p["fps"].astype(float))
        res = {"n_ab": int(n_ab.sum())}
        res["zone_rate"] = _ci(zone_n[idx].sum(1), n_p[idx].sum(1), zone_n.sum() / n_p.sum())
        res["first_pitch_strike"] = _ci(fps_n[idx].sum(1), n_ab[idx].sum(1), fps_n.sum() / n_ab.sum())
        res["pitches_per_ab"] = _ci(n_p[idx].sum(1), n_ab[idx].sum(1), n_p.sum() / n_ab.sum())
        res["outcomes"] = {}
        for o in ("strikeout", "walk", "in_play"):
            k = per(a[a["end"] == o])
            res["outcomes"][o] = _ci(k[idx].sum(1), n_ab[idx].sum(1), k.sum() / n_ab.sum())
        # mix by count, renormalised over the bank types (excluded types are absent from the sim)
        pt = p[p["pitch_type"].isin(bank_types)]
        M = np.zeros((G, len(COUNTS), len(bank_types)))
        np.add.at(M, (pt["game_pk"].map(gi).to_numpy(), pt["count"].map(COUNTS.index).to_numpy(),
                      pt["pitch_type"].map(ti).to_numpy()), 1.0)
        B, tot_c = M[idx].sum(1), M.sum(0)                  # (n_boot, counts, types), (counts, types)
        res["mix_by_count"], res["mix_n"], res["mix_tv95"] = {}, {}, {}
        for ci, c in enumerate(COUNTS):
            if tot_c[ci].sum() == 0:
                continue
            est = tot_c[ci] / tot_c[ci].sum()
            bs = B[:, ci] / np.maximum(B[:, ci].sum(1, keepdims=True), 1)
            res["mix_by_count"][c] = dict(zip(bank_types, est.tolist()))
            res["mix_n"][c] = int(tot_c[ci].sum())
            # TV distance of a resampled mix from the point mix: how far sampling noise alone moves it
            res["mix_tv95"][c] = float(np.percentile(0.5 * np.abs(bs - est).sum(1), 95))
        out[s] = res
    return out


def load_test(slug: str):
    """(held-out games of the pooled-season frame, seasons); same prep as m7_export.load_real's pooled."""
    raw = load_seasons(sorted(RAW_DIR.glob(f"{SLUGS[slug][0]}_*.csv")))
    pooled = add_location(add_context(raw).dropna(subset=["pitch_type"]))
    return split_by_game(pooled, seed=0)[1], sorted(int(y) for y in raw["game_year"].unique())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("slug", nargs="?")
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for slug in (SLUGS if a.all else [a.slug]):
        face = json.loads((FACE_DIR / f"{slug}.json").read_text())
        test, seasons = load_test(slug)
        r = real_stats(test, sorted(face["types"]), face["zone"])
        r.update(slug=slug, seasons=seasons, excluded_types=face.get("excluded", []))
        (OUT / f"{slug}-real.json").write_text(json.dumps(r, indent=1))
        print(slug, r["n_ab_total"], "ABs", r["excluded_abs"], f"excluded-type share {r['excluded_type_share']:.3f}")


if __name__ == "__main__":
    sys.exit(main())
