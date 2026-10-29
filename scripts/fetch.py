"""
Download every pitch a pitcher threw in a date range from Baseball Savant.

Examples:
    python scripts/fetch.py --name "Tarik Skubal" --start 2025-04-01 --end 2025-09-30
    python scripts/fetch.py --id 669373 --start 2025-04-01 --end 2025-09-30

Saves to data/<mlbam_id>_<start>_<end>.csv and prints a quick summary.
"""

import argparse
from pathlib import Path

from pybaseball import cache, playerid_lookup, statcast_pitcher

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def lookup_id(full_name: str) -> int:
    first, last = full_name.strip().split(" ", 1)
    res = playerid_lookup(last, first)
    res = res.dropna(subset=["key_mlbam"])
    if res.empty:
        raise SystemExit(f"No MLB player found for '{full_name}'")
    if len(res) > 1:
        print("Several matches, using the most recent:\n",
              res[["name_first", "name_last", "key_mlbam", "mlb_played_last"]])
        res = res.sort_values("mlb_played_last", ascending=False)
    return int(res.iloc[0]["key_mlbam"])


def main():
    ap = argparse.ArgumentParser()
    who = ap.add_mutually_exclusive_group(required=True)
    who.add_argument("--name", help='e.g. "Tarik Skubal"')
    who.add_argument("--id", type=int, help="MLBAM player id")
    ap.add_argument("--start", required=True, help="YYYY-MM-DD")
    ap.add_argument("--end", required=True, help="YYYY-MM-DD")
    args = ap.parse_args()

    cache.enable()
    pid = args.id or lookup_id(args.name)
    df = statcast_pitcher(args.start, args.end, pid)
    if df.empty:
        raise SystemExit("No pitches returned; check the id and date range.")

    DATA_DIR.mkdir(exist_ok=True)
    path = DATA_DIR / f"{pid}_{args.start}_{args.end}.csv"
    df.to_csv(path, index=False)

    print(f"Saved {len(df):,} pitches to {path}")
    summary = (df.groupby("pitch_name")
                 .agg(n=("pitch_type", "size"),
                      mph=("release_speed", "mean"),
                      rpm=("release_spin_rate", "mean"),
                      axis=("spin_axis", "mean"))
                 .sort_values("n", ascending=False).round(1))
    print(summary.to_string())


if __name__ == "__main__":
    main()
