"""Usable-clip rate per home venue, from every data/video/<slug>/pose2d.csv (M5).

usage: python scripts/m5_venue_yield.py [--out data/video/venue_yield.csv]
The home broadcast decides whether the centre-field view is used, so this is what predicts a good clip.
"""
import argparse
import glob
from pathlib import Path

import pandas as pd

from pitchsim.pose.clips import venue_yield_table


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="data/video")
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data/video/venue_yield.csv")
    a = ap.parse_args()
    cols = ["game_pk", "home_team"]  # some raw CSVs (non-Statcast extras) lack home_team: skip them
    raw = pd.concat([pd.read_csv(f, usecols=cols) for f in glob.glob(f"{a.raw}/*.csv")
                     if set(cols) <= set(pd.read_csv(f, nrows=0).columns)])
    home = raw.drop_duplicates("game_pk").set_index("game_pk")["home_team"]
    rows = []
    for d in Path(a.video).iterdir():
        if not (d / "pose2d.csv").exists():
            continue
        pose = pd.read_csv(d / "pose2d.csv")
        pose["game_pk"] = pose["clip"].str.split("_").str[0].astype(int)
        rows.append(pose[["game_pk", "ok"]].assign(ok=pose["ok"].astype(bool)))
    df = pd.concat(rows)
    df["home_team"] = df["game_pk"].map(home)
    df = df.dropna(subset=["home_team"])
    t = venue_yield_table(df)
    t.to_csv(a.out, index=False)
    print(f"overall usable {df['ok'].mean():.2f} over {len(df)} clips")
    print(t.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
