"""
Download every pitch a pitcher threw in a date range from Baseball Savant.

Examples:
    python scripts/fetch.py --name "Tarik Skubal" --start 2025-04-01 --end 2025-09-30
    python scripts/fetch.py --id 669373 --start 2025-04-01 --end 2025-09-30

Saves to data/raw/<mlbam_id>_<start>_<end>.csv and prints a quick summary.
"""

import argparse

import pandas as pd

from pitchsim.data import fetch_pitcher


def main():
    ap = argparse.ArgumentParser()
    who = ap.add_mutually_exclusive_group(required=True)
    who.add_argument("--name", help='e.g. "Tarik Skubal"')
    who.add_argument("--id", type=int, help="MLBAM player id")
    ap.add_argument("--start", required=True, help="YYYY-MM-DD")
    ap.add_argument("--end", required=True, help="YYYY-MM-DD")
    args = ap.parse_args()

    path = fetch_pitcher(args.name, args.id, args.start, args.end)
    df = pd.read_csv(path)
    print(f"{len(df):,} pitches in {path}")
    summary = (df.groupby("pitch_name")
                 .agg(n=("pitch_type", "size"),
                      mph=("release_speed", "mean"),
                      rpm=("release_spin_rate", "mean"),
                      spin_axis=("spin_axis", "mean"))
                 .sort_values("n", ascending=False).round(1))
    print(summary.to_string())


if __name__ == "__main__":
    main()
