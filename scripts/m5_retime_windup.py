"""Opt-in windup speed-up of one pitcher's finished delivery JSON (pitchsim.pose.anchor.retime_windup).

usage: python scripts/m5_retime_windup.py IN.json --leg-lift 117 --factor 0.90 --out OUT.json

The windup before the leg-lift peak is replayed in factor * leg_lift frames; the tail (peak, stride, release)
is copied bit for bit, so the delivery is `removed` frames shorter and release_index / pre_s shift by that many
frames (the viewer places the pitcher by release_index + t * fps, so nothing else depends on the count).
Refuses to run twice (metadata "retime_windup").  --out is required: never overwrites IN.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from pitchsim.pose.anchor import retime_windup, segment_lengths

ap = argparse.ArgumentParser()
ap.add_argument("delivery")
ap.add_argument("--leg-lift", type=int, required=True)
ap.add_argument("--factor", type=float, required=True, help="new/old windup length, e.g. 0.90 = 10%% faster")
ap.add_argument("--out", required=True)
a = ap.parse_args()
d = json.loads(Path(a.delivery).read_text())
if "retime_windup" in d:
    raise SystemExit("already retimed")
j = np.array(d["joints_ft"])
M = int(round(a.leg_lift * a.factor))
o = retime_windup(j, d["fps"], a.leg_lift, M, d["throws"])
rm = a.leg_lift - M
assert np.array_equal(np.round(o[M:], 3), j[a.leg_lift:])
d["joints_ft"] = np.round(o, 3).tolist()
d["release_index"] -= rm
d["pre_s"] = round(d["release_index"] / d["fps"], 4)
d["retime_windup"] = {"old_leg_lift_index": a.leg_lift, "leg_lift_index": M, "frames_removed": rm, "factor": M / a.leg_lift,
                      "max_bone_change_ft": float(np.abs(segment_lengths(o[M:]) - segment_lengths(j[a.leg_lift:])).max())}
Path(a.out).write_text(json.dumps(d))
print(d["retime_windup"], d["release_index"], d["pre_s"], len(d["joints_ft"]))
