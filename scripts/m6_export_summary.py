"""Copy M6's held-out numbers into a tracked file (results/ is gitignored) for the MCP server.

    python scripts/m6_export_summary.py [results_dir]   # default: results/

Reads m6_folds.csv, m6_transformer_folds.csv, m6_reliability.csv (all written by the M6 scripts)
and writes src/pitchsim/sequence/m6_held_out.json: per pitcher, pitch-weighted mean held-out log
loss per model (5 game folds) and the largest calibration-bin gap (bins with >= 100 forecasts).
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
src = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results"
folds = pd.read_csv(src / "m6_folds.csv")
tf = pd.read_csv(src / "m6_transformer_folds.csv")
rel = pd.read_csv(src / "m6_reliability.csv")


def wmean(g: pd.DataFrame, col: str) -> float:
    return round(float((g[col] * g["test_pitches"]).sum() / g["test_pitches"].sum()), 3)


out = {}
for name, g in folds.groupby("pitcher", sort=False):
    ll = {m: wmean(g, f"logloss_{m}") for m in
          ["mix", "count", "logistic", "logistic_plus_tto", "boosted"]}
    t = tf[tf["pitcher"] == name]   # transformer covers the original three only
    ll["transformer"] = wmean(t, "transformer") if len(t) else None
    r = rel[(rel["pitcher"] == name) & (rel["model"] == "logistic") & (rel["n"] >= 100)]
    out[name] = {"n_pitches": int(g["test_pitches"].sum()), "folds": len(g),
                 "held_out_log_loss": ll,
                 "logistic_max_calibration_gap": round(float((r["mean_pred"] - r["observed"]).abs().max()), 3)}
dest = ROOT / "src" / "pitchsim" / "sequence" / "m6_held_out.json"
dest.write_text(json.dumps(out, indent=1), encoding="utf-8")
print(json.dumps(out, indent=1))
