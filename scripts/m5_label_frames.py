"""Contact sheets around each detected release, and detector error against hand labels.
usage: python scripts/m5_label_frames.py sheets data/video/ohtani-shohei [clip ...]
       python scripts/m5_label_frames.py score data/video/ohtani-shohei
Sheets are images: they stay in data/ (gitignored)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

LABELS = Path(__file__).parents[1] / "src/pitchsim/pose/labels/ohtani_release_frames.csv"


def sheets(d: Path, only: list[str]):
    """Each arg is `clip.mp4` (+-6 frames around the detected release), `clip.mp4@N` (+-6 around
    frame N) or `clip.mp4@N:S` (13 frames around N, S frames apart; S=12 gives an overview);
    append `:CX:CY` to crop around a fixed pixel instead of the tracked hips (when tracking failed)."""
    import cv2
    from pitchsim.pose.pose2d import load_pose2d
    det = pd.read_csv(d / "pose2d.csv").set_index("clip")
    (d / "labels").mkdir(exist_ok=True)
    for arg in only:
        clip, _, rest = arg.partition("@")
        centre, step, *fixed = (rest.split(":") + ["", ""])[:2] + rest.split(":")[2:]
        centre, step = (int(centre) if centre else int(det.loc[clip, "release_frame"])), int(step or 1)
        pose = load_pose2d(d / "pose2d" / clip.replace(".mp4", ".npz"))
        cap = cv2.VideoCapture(str(d / clip))
        tiles = []
        for f in range(centre - 6 * step, centre + 6 * step + 1, step):
            if f < 0 or f >= len(pose.xy):
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, im = cap.read()
            if not ok:
                continue
            # crop a 300 px square around the pitcher's hips so the hand is big enough to judge
            cx, cy = (int(fixed[0]), int(fixed[1])) if fixed else pose.xy[f, [11, 12]].mean(0).astype(int)
            cx, cy = np.clip(cx, 150, im.shape[1] - 150), np.clip(cy, 150, im.shape[0] - 150)
            im = cv2.resize(im[cy - 150: cy + 150, cx - 150: cx + 150], (340, 340))
            cv2.putText(im, str(f), (8, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 255), 3)
            tiles.append(im)
        cap.release()
        while len(tiles) % 4:
            tiles.append(np.zeros_like(tiles[0]))
        grid = np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)])
        out = clip.replace(".mp4", f"_{centre}_{step}.jpg")
        cv2.imwrite(str(d / "labels" / out), grid, [cv2.IMWRITE_JPEG_QUALITY, 80])


def score(d: Path):
    """Detector error vs hand labels.  fresh=no rows are IN-SAMPLE: the release rule (first peak
    >= 0.5 x max) was chosen after seeing those labels.  fresh=yes rows were labelled after."""
    det = pd.read_csv(d / "pose2d.csv").set_index("clip")
    lab = pd.read_csv(LABELS)
    lab["err"] = det.loc[lab["clip"], "release_frame"].to_numpy() - lab["release_frame_label"].to_numpy()
    for name, g in (("IN-SAMPLE (rule chosen after seeing these labels)", lab[lab["fresh"] == "no"]),
                    ("FRESH (labelled after the rule was fixed)", lab[lab["fresh"] == "yes"])):
        e = g["err"]
        print(f"{name}: n={len(g)} median |err|={np.median(np.abs(e))} max |err|={np.abs(e).max()} "
              f"within 2 frames={(np.abs(e) <= 2).sum()}/{len(g)}")
        print(g[["clip", "release_frame_label", "err"]].to_string(index=False))


if __name__ == "__main__":
    d = Path(sys.argv[2])
    sheets(d, sys.argv[3:]) if sys.argv[1] == "sheets" else score(d)
