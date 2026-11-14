"""Run 2D pose on every ok/cached clip, screen it, find the release frame.
usage: python scripts/m5_pose2d.py data/video/ohtani-shohei/clips.csv [shard n_shards]
Shards only fill the cache (CPU inference is slow: run several in parallel, then one plain run).
Writes pose2d.csv next to clips.csv and caches keypoints in pose2d/<clip>.npz."""
import sys
from pathlib import Path

import pandas as pd

from pitchsim.pose.clips import clip_name
from pitchsim.pose.pose2d import load_pose2d, run_pose2d, save_pose2d
from pitchsim.pose.release import release_frame
from pitchsim.pose.screen import screen

OVERRIDES = Path(__file__).parents[1] / "src/pitchsim/pose/labels/ohtani_screen_overrides.csv"

csv = Path(sys.argv[1])
df = pd.read_csv(csv)
df = df[df["status"].isin(["ok", "cached"])]
# Hand review beats the automatic screen: clips I looked at and found to be the wrong view.
over = pd.read_csv(OVERRIDES).set_index("clip")["reason"].to_dict() if OVERRIDES.exists() else {}
rows = []
shard, n_shards = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else (0, 1)
for k, (_, r) in enumerate(df.iterrows()):
    if k % n_shards != shard:
        continue
    name = clip_name(r)
    npz = csv.parent / "pose2d" / name.replace(".mp4", ".npz")
    if npz.exists():   # cached: never redo inference
        p = load_pose2d(npz)
    else:
        p = run_pose2d(r["clip_path"])
        save_pose2d(p, npz)
    ok, reason = screen(p, r["p_throws"])
    if name in over:
        ok, reason = False, over[name]
    rows.append(dict(clip=name, ok=ok, reason=reason, release_frame=release_frame(p, r["p_throws"]),
                     fps=p.fps, n_frames=len(p.xy), split=r["split"]))
    print(rows[-1], flush=True)
out = pd.DataFrame(rows)
out.to_csv(csv.parent / "pose2d.csv", index=False)
out["reason"] = out["reason"].fillna("")
print(pd.crosstab(out["reason"].replace("", "ok"), out["split"], margins=True))
