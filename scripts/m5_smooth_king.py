"""Opt-in smoothing of King's finished delivery JSON (pitchsim.pose.king_smooth).

usage: python scripts/m5_smooth_king.py IN.json --out OUT.json [--leg-lift 123]

Never overwrites its input: write to a scratch --out, check it, then copy it over by hand.  Refuses to run
twice (metadata "smooth_king").  Only the windup (before the leg-lift peak + 4 frames) and the upper body in the
last 0.4 s before release move; release frame, planted feet and bone lengths are exact.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from pitchsim.pose.anchor import segment_lengths
from pitchsim.pose.king_smooth import smooth_king

ap = argparse.ArgumentParser()
ap.add_argument("delivery")
ap.add_argument("--out", required=True)
ap.add_argument("--leg-lift", type=int, default=123, help="leg-lift index (results/m5/king-michael/correction.json)")
a = ap.parse_args()
if Path(a.out).resolve() == Path(a.delivery).resolve():
    raise SystemExit("--out must differ from the input")
d = json.loads(Path(a.delivery).read_text())
if "smooth_king" in d:
    raise SystemExit("already smoothed")
j = np.array(d["joints_ft"])
o = smooth_king(j, d["fps"], a.leg_lift, d["release_index"], d["throws"])
d["smooth_king"] = {"leg_lift_index": a.leg_lift,
                    "max_joint_move_ft": float(np.linalg.norm(o - j, axis=2).max()),
                    "release_pose_delta_ft": float(np.abs(o[d["release_index"]] - j[d["release_index"]]).max()),
                    "max_bone_change_ft": float(np.abs(segment_lengths(o) - segment_lengths(j)).max())}
d["joints_ft"] = np.round(o, 3).tolist()
Path(a.out).write_text(json.dumps(d))
print(d["smooth_king"])
