"""Opt-in start-of-windup lead-knee fix of one pitcher's finished delivery JSON (pitchsim.pose.anchor.fix_lead_stance).

usage: python scripts/m5_fix_lead_stance.py IN.json --until 50 [--out OUT.json]

Post-processes the shipped delivery (no m5_build re-run); frames >= --until are bit-identical.
Refuses to run twice (metadata "fix_lead_stance").  Without --out it overwrites IN.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from pitchsim.pose.anchor import fix_lead_stance, segment_lengths

ap = argparse.ArgumentParser()
ap.add_argument("delivery")
ap.add_argument("--until", type=int, required=True)
ap.add_argument("--out")
a = ap.parse_args()
d = json.loads(Path(a.delivery).read_text())
if "fix_lead_stance" in d:
    raise SystemExit("already fixed")
j = np.array(d["joints_ft"])
o = fix_lead_stance(j, a.until, d["throws"])
d["fix_lead_stance"] = {"until_index": a.until,
                        "max_joint_move_ft": float(np.linalg.norm(o - j, axis=2).max()),
                        "max_bone_change_ft": float(np.abs(segment_lengths(o) - segment_lengths(j)).max())}
d["joints_ft"] = np.round(o, 3).tolist()
Path(a.out or a.delivery).write_text(json.dumps(d))
print(d["fix_lead_stance"])
