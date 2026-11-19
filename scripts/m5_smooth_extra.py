"""Opt-in final windup smoothing of one pitcher's finished delivery JSON (pitchsim.pose.anchor.smooth_extra).

usage: python scripts/m5_smooth_extra.py web/public/data/deliveries/cease-dylan.json --leg-lift 117

Post-processes the shipped delivery instead of re-running m5_build, so nothing upstream (arm, correction, clips)
is recomputed; only joints before the leg-lift peak move.  Refuses to run twice (metadata "smooth_extra").
Only run for a pitcher that asks for it; every other delivery is untouched.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from pitchsim.pose.anchor import segment_lengths, smooth_extra

SIGMA_S, BLEND_S = 0.10, 0.4  # sigma 0.10 s halves Cease's lead-ankle spike at <=0.15 ft joint move; wider only moves the foot more

ap = argparse.ArgumentParser()
ap.add_argument("delivery")
ap.add_argument("--leg-lift", type=int, required=True, help="leg-lift index (results/m5/<slug>/correction.json)")
a = ap.parse_args()
p = Path(a.delivery)
d = json.loads(p.read_text())
if "smooth_extra" in d:
    raise SystemExit("already smoothed")
j = np.array(d["joints_ft"])
o = smooth_extra(j, d["fps"], a.leg_lift, d["throws"], SIGMA_S, BLEND_S)
d["smooth_extra"] = {"sigma_s": SIGMA_S, "blend_s": BLEND_S, "until_index": a.leg_lift,
                     "max_joint_move_ft": float(np.linalg.norm(o - j, axis=2).max()),
                     "max_bone_change_ft": float(np.abs(segment_lengths(o) - segment_lengths(j)).max())}
d["joints_ft"] = np.round(o, 3).tolist()
p.write_text(json.dumps(d))
print(d["smooth_extra"])
