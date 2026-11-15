"""Fetch Savant broadcast clips for a pitcher's bases-empty pitches (M5).

usage: python scripts/m5_fetch_clips.py CSV --out data/video/<slug> [--max-train 120 --max-test 80]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from pitchsim.pose.clips import KEY, bases_empty, fetch_clips
from pitchsim.statcast import clean
from pitchsim.validation import split_by_game

META = ["pitch_type", "release_pos_x", "release_pos_z", "release_extension", "arm_angle", "p_throws"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--max-train", type=int, default=120)
    ap.add_argument("--max-test", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-download", action="store_true",
                    help="resolve play ids and write clips.csv only; request no mp4 (the host 403s scripts)")
    a = ap.parse_args()

    df = clean(pd.read_csv(a.csv))
    rng = np.random.default_rng(a.seed)
    # split by game on all cleaned rows (same games as the rest of M5), then keep bases empty
    train, test = (p[bases_empty(p)] for p in split_by_game(df, seed=0))
    parts = []
    for name, part, cap in (("train", train, a.max_train), ("test", test, a.max_test)):
        pick = part.iloc[np.sort(rng.permutation(len(part))[:cap])].copy()
        pick["split"] = name
        parts.append(pick)
    sel = pd.concat(parts)
    res = fetch_clips(sel, Path(a.out), download=not a.no_download)
    keep = sel[KEY + ["split"] + META].reset_index(drop=True)
    res = res.merge(keep, on=KEY, how="left")
    res.to_csv(Path(a.out) / "clips.csv", index=False)
    print(res["status"].value_counts().to_string())


if __name__ == "__main__":
    main()
