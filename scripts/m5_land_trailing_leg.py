"""Opt-in follow-through fix of one pitcher's finished delivery JSON (pitchsim.pose.anchor.land_trailing_leg).

usage: python scripts/m5_land_trailing_leg.py IN.json --start 166 --land 188 --out OUT.json

Frames <= max(release_index, --start) are bit-identical.  Refuses to run twice (metadata "land_trailing_leg").
"""
import argparse
import json
from pathlib import Path

import numpy as np

from pitchsim.pose.anchor import land_trailing_leg, segment_lengths

ap = argparse.ArgumentParser()
ap.add_argument("delivery")
ap.add_argument("--start", type=int, required=True)
ap.add_argument("--land", type=int, required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
d = json.loads(Path(a.delivery).read_text())
if "land_trailing_leg" in d:
    raise SystemExit("already done")
j = np.array(d["joints_ft"])
o = land_trailing_leg(j, d["release_index"], a.start, a.land, d["throws"])
d["land_trailing_leg"] = {"start_index": a.start, "land_index": a.land,
                          "max_joint_move_ft": float(np.linalg.norm(o - j, axis=2).max())}
d["joints_ft"] = np.round(o, 3).tolist()
Path(a.out).write_text(json.dumps(d))
print(d["land_trailing_leg"])
