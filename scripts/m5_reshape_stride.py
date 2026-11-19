"""Opt-in lead-knee stride reshape of one pitcher's finished delivery JSON (pitchsim.pose.stride_reshape).

usage: python scripts/m5_reshape_stride.py IN.json --start 128 --until 150 --fall-from 137 [--lam 30 --fall-weight 0.03] [--out OUT.json]

Frames outside [start, until] are bit-identical.  Refuses to run twice (metadata "reshape_stride").
Without --out it overwrites IN.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from pitchsim.pose.anchor import segment_lengths
from pitchsim.pose.stride_reshape import knee_angle_deg, reshape_stride

ap = argparse.ArgumentParser()
ap.add_argument("delivery")
ap.add_argument("--start", type=int, required=True)
ap.add_argument("--until", type=int, required=True)
ap.add_argument("--fall-from", type=int, required=True)
ap.add_argument("--lam", type=float, default=30.0)
ap.add_argument("--fall-weight", type=float, default=0.1)
ap.add_argument("--out")
a = ap.parse_args()
d = json.loads(Path(a.delivery).read_text())
if "reshape_stride" in d:
    raise SystemExit("already reshaped")
j = np.array(d["joints_ft"])
o = reshape_stride(j, a.start, a.until, d["throws"], a.lam, a.fall_from, a.fall_weight)
lead = "L" if d["throws"] == "R" else "R"
ang = lambda x: knee_angle_deg(x, lead)
d["reshape_stride"] = {"start_index": a.start, "until_index": a.until, "fall_from": a.fall_from,
                       "lam": a.lam, "fall_weight": a.fall_weight,
                       "max_joint_move_ft": float(np.linalg.norm(o - j, axis=2).max()),
                       "max_bone_change_ft": float(np.abs(segment_lengths(o) - segment_lengths(j)).max()),
                       "peak_knee_deg_per_s": [float(np.abs(np.gradient(ang(x))[a.start - 4:a.until + 1]).max() * 60) for x in (j, o)]}
d["joints_ft"] = np.round(o, 3).tolist()
Path(a.out or a.delivery).write_text(json.dumps(d))
print(d["reshape_stride"])
