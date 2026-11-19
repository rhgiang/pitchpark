"""Opt-in lead-knee buckle fix of one pitcher's finished delivery JSON (pitchsim.pose.anchor.smooth_knee).

usage: python scripts/m5_smooth_knee.py IN.json --leg-lift 117 --start 90 [--out OUT.json]

Post-processes the shipped delivery (no m5_build re-run); frames >= --leg-lift are bit-identical.
Refuses to run twice (metadata "smooth_knee").  Without --out it overwrites IN.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from pitchsim.pose.anchor import segment_lengths, smooth_knee

ap = argparse.ArgumentParser()
ap.add_argument("delivery")
ap.add_argument("--leg-lift", type=int, required=True)
ap.add_argument("--start", type=int, required=True)
ap.add_argument("--out")
a = ap.parse_args()
d = json.loads(Path(a.delivery).read_text())
if "smooth_knee" in d:
    raise SystemExit("already smoothed")
j = np.array(d["joints_ft"])
o = smooth_knee(j, d["fps"], a.start, a.leg_lift, d["throws"])
d["smooth_knee"] = {"start_index": a.start, "until_index": a.leg_lift,
                    "max_joint_move_ft": float(np.linalg.norm(o - j, axis=2).max()),
                    "max_bone_change_ft": float(np.abs(segment_lengths(o) - segment_lengths(j)).max())}
d["joints_ft"] = np.round(o, 3).tolist()
Path(a.out or a.delivery).write_text(json.dumps(d))
print(d["smooth_knee"])
