"""Rename hand-saved Savant clips to <game_pk>_<at_bat>_<pitch>.mp4 (M5).  Safe to re-run.

usage: python scripts/m5_match_clips.py data/video/<slug>
For each row of download_list.csv: fetch its sporty-videos page (cached in <dir>/pages/), take the mp4
basename Savant saves it under, rename the matching file.  Never deletes; reports what is left over.
Then marks found clips `cached` in clips.csv.
"""
import argparse
from pathlib import Path

import pandas as pd

from pitchsim.pose.clips import _get_page, clip_name, clip_url_from_page, match_downloads, mp4_basename


def page(d: Path, pid: str) -> str:
    f = d / "pages" / f"{pid}.html"
    if not f.exists():
        f.parent.mkdir(exist_ok=True)
        f.write_text(_get_page(pid), encoding="utf-8")
    return f.read_text(encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("clips_dir")
    d = Path(ap.parse_args().clips_dir)
    want = pd.read_csv(d / "download_list.csv")
    expected = {}
    for _, r in want.iterrows():
        url = clip_url_from_page(page(d, r["play_id"]))
        expected[clip_name(r)] = mp4_basename(url) if url else None
    res = match_downloads(expected, [p.name for p in d.iterdir() if p.is_file()])
    for saved, name in res["renames"].items():
        (d / saved).rename(d / name)
        print("renamed", saved, "->", name)
    found = set(res["have"]) | set(res["renames"].values())
    print(f"{len(found)}/{len(expected)} listed clips in place")
    for k in ("missing", "duplicates", "partial", "leftovers"):
        print(f"{k} ({len(res[k])}):", res[k])
    cl = pd.read_csv(d / "clips.csv")
    ok = pd.Series([clip_name(r) in found for _, r in cl.iterrows()], index=cl.index)
    cl.loc[ok, "status"] = "cached"
    cl["clip_path"] = cl["clip_path"].astype(object)  # all-empty columns read back as float
    cl.loc[ok, "clip_path"] = [str(d / clip_name(r)) for _, r in cl[ok].iterrows()]
    cl.to_csv(d / "clips.csv", index=False)


if __name__ == "__main__":
    main()
