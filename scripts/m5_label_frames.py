"""Contact sheets around each detected release, and detector error against hand labels.
usage: python scripts/m5_label_frames.py sheets data/video/ohtani-shohei [clip ...]
       python scripts/m5_label_frames.py score data/video/ohtani-shohei
       python scripts/m5_label_frames.py fssheets data/video/ohtani-shohei [clip ...]   (foot strike)
       python scripts/m5_label_frames.py fsscore data/video/ohtani-shohei
Sheets are images: they stay in data/ (gitignored)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _paths(clips_dir):
    from pitchsim.pose.clips import label_path
    slug = Path(clips_dir).name
    return label_path(slug, "release_frames"), label_path(slug, "foot_strike_frames")


def sheets(d: Path, only: list[str]):
    """Each arg is `clip.mp4` (+-6 frames around the detected release), `clip.mp4@N` (+-6 around
    frame N) or `clip.mp4@N:S` (13 frames around N, S frames apart; S=12 gives an overview);
    append `:CX:CY` to crop around a fixed pixel instead of the tracked hips (when tracking failed)."""
    import cv2
    from pitchsim.pose.pose2d import load_pose2d
    (d / "labels").mkdir(exist_ok=True)
    for arg in only:
        clip, _, rest = arg.partition("@")
        centre, step, *fixed = (rest.split(":") + ["", ""])[:2] + rest.split(":")[2:]
        # pose2d.csv only when no frame is given (clip@N works while m5_pose2d is still running)
        centre = int(centre) if centre else int(pd.read_csv(d / "pose2d.csv").set_index("clip").loc[clip, "release_frame"])
        step = int(step or 1)
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
    lab = pd.read_csv(_paths(d)[0])
    lab["err"] = det.loc[lab["clip"], "release_frame"].to_numpy() - lab["release_frame_label"].to_numpy()
    for name, g in (("IN-SAMPLE (rule chosen after seeing these labels)", lab[lab["fresh"] == "no"]),
                    ("FRESH (labelled after the rule was fixed)", lab[lab["fresh"] == "yes"])):
        e = g["err"]
        print(f"{name}: n={len(g)} median |err|={np.median(np.abs(e))} max |err|={np.abs(e).max()} "
              f"within 2 frames={(np.abs(e) <= 2).sum()}/{len(g)}")
        print(g[["clip", "release_frame_label", "err"]].to_string(index=False))


def _throws(d: Path) -> str:
    """The pitcher's hand from clips.csv (a LHP's lead foot is his RIGHT one)."""
    return str(pd.read_csv(d / "clips.csv")["p_throws"].iloc[0])


def release_pose(d: Path, clip: str) -> int:
    """Release pose frame: release hand label - 1 where labelled (as m5_build), else the detector's."""
    lab = pd.read_csv(_paths(d)[0]).set_index("clip")
    if clip in lab.index:
        return int(lab.loc[clip, "release_frame_label"]) - 1
    return int(pd.read_csv(d / "pose2d.csv").set_index("clip").loc[clip, "release_frame"])


def fs_sheets(d: Path, only: list[str]):
    """Foot-strike sheets: frames release-22 .. release-3, one apart, a FIXED 140 px crop around the lead
    ankle at release (a fixed crop shows the foot stopping; a hip-centred one moves with the body), the
    2D lead ankle (COCO 15, or 16 for a LHP) dotted so it is clear which foot is the lead.  Label rule: first frame the
    lead foot is flat on the ground (toe down) and stays put."""
    import cv2
    from pitchsim.pose.pose2d import load_pose2d
    from pitchsim.pose.release import lead_ankle_idx
    k = lead_ankle_idx(_throws(d))
    (d / "labels").mkdir(exist_ok=True)
    for clip in only:
        r = release_pose(d, clip)
        p = load_pose2d(d / "pose2d" / clip.replace(".mp4", ".npz"))
        cx, cy = p.xy[r, k].astype(int) - [0, 25]
        cap, tiles = cv2.VideoCapture(str(d / clip)), []
        for f in range(r - 22, r - 2):
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, im = cap.read()
            if not ok:
                continue
            cv2.circle(im, tuple(int(v) for v in p.xy[f, k]), 3, (0, 255, 0), -1)
            x, y = np.clip(cx, 70, im.shape[1] - 70), np.clip(cy, 70, im.shape[0] - 70)
            im = cv2.resize(im[y - 70: y + 70, x - 70: x + 70], (340, 340))
            cv2.putText(im, f"{f} r{f - r}", (8, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 3)
            tiles.append(im)
        cap.release()
        while len(tiles) % 5:
            tiles.append(np.zeros_like(tiles[0]))
        grid = np.vstack([np.hstack(tiles[i:i + 5]) for i in range(0, len(tiles), 5)])
        cv2.imwrite(str(d / "labels" / f"fs_{clip.replace('.mp4', '')}.jpg"), grid, [cv2.IMWRITE_JPEG_QUALITY, 80])


def fs_score(d: Path):
    """Foot-strike detector (release.foot_strike_frame) vs hand labels.  In-sample: its tolerance was
    picked on these labels."""
    from pitchsim.pose.pose2d import load_pose2d
    from pitchsim.pose.release import foot_strike_frame
    lab = pd.read_csv(_paths(d)[1])
    rows = []
    for c, lf in zip(lab["clip"], lab["foot_strike_frame_label"]):
        r = release_pose(d, c)
        f = foot_strike_frame(load_pose2d(d / "pose2d" / c.replace(".mp4", ".npz")), _throws(d), r)
        rows.append({"clip": c, "label": lf, "label_to_release": r - lf, "detected": f,
                     "err": None if f is None else f - lf})
    t = pd.DataFrame(rows)
    e = t["err"].dropna().abs()
    print(t.to_string(index=False))
    print(f"n={len(t)} found={len(e)} median |err|={e.median()} max |err|={e.max()} "
          f"within 2={(e <= 2).sum()}/{len(t)}; label -> release median {t['label_to_release'].median()} frames")


if __name__ == "__main__":
    d = Path(sys.argv[2])
    {"sheets": lambda: sheets(d, sys.argv[3:]), "score": lambda: score(d),
     "fssheets": lambda: fs_sheets(d, sys.argv[3:]), "fsscore": lambda: fs_score(d)}[sys.argv[1]]()
