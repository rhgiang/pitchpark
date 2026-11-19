"""M5: the throwing arm's image-plane angle through the arm action, real 2D keypoints vs the replica (Sale round 2).

usage: python scripts/m5_arm_curve.py data/video/sale-chris [--tag before] [--overlay 824873_21_3.mp4 ...]
Needs a built delivery + m5_compare2d.py run (the replica's fitted camera per clip is in <slug>.compare.json).

For every screened-ok clip and every frame from -0.40 s to +0.15 s around release, the angle of shoulder -> wrist
(and shoulder -> elbow) in the image, in degrees above horizontal toward the throwing side (the catcher's-view
arm-angle convention of Statcast), from (a) the clip's RTMPose keypoints and (b) the replica projected through the
clip's fitted camera.  Error = replica - real (positive: the replica's arm is higher).  Frames count only where
shoulder and the joint are both confident (>= CONF_MIN).  Writes results/m5/<slug>/arm_curve_<tag>.csv and prints
the median curve by frame offset from release; --overlay draws the replica arm on the real frames.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from pitchsim.pose.compare2d import (COCO_IDX, CONF_MIN, SMPL_IDX, image_arm_angle_deg, project, sample_frames,
                                     shot_start)
from pitchsim.pose.pose2d import load_pose2d

ROOT = Path(__file__).resolve().parents[1]
COCO = {"L": (5, 7, 9), "R": (6, 8, 10)}      # COCO-17 shoulder, elbow, wrist
OFFSETS_S = (-0.15, -0.08, -0.03, 0.0, 0.05)  # overlay stills
PRE_S, POST_S = 0.40, 0.15


def wrap(d):
    return (np.asarray(d) + 180.0) % 360.0 - 180.0


def clip_curves(c, vd, cm, clips, rep, ri, fps_d, throws):
    """Per-frame table for one clip: t, real/replica angles of wrist and elbow, plus the projected points."""
    p = load_pose2d(vd / "pose2d" / c.replace(".mp4", ".npz"))
    rp, fps = clips[c]["release_pose_frame"], clips[c]["fps"]
    xy = p.xy.astype(float)
    cut = shot_start(xy[:, COCO_IDX], p.conf[:, COCO_IDX] >= CONF_MIN, rp, fps)
    f = np.arange(int(rp - PRE_S * fps), int(rp + POST_S * fps) + 1)
    f = f[(f >= max(cut, cm["frame0"])) & (f < len(xy))]
    t = (f - rp) / fps
    A = np.array(cm["A"])
    b = np.array([np.nan * np.ones(2) if cm["b"][i - cm["frame0"]] is None else cm["b"][i - cm["frame0"]] for i in f])
    X = sample_frames(rep, ri + t * fps_d)
    P = project(A, np.nan_to_num(b), X)           # (F, 12, 2)
    P[np.isnan(b[:, 0])] = np.nan
    s_c, e_c, w_c = COCO[throws]
    s_r, e_r, w_r = (0, 2, 4) if throws == "L" else (1, 3, 5)   # positions in the 12-joint arrays
    side = 1 if np.nanmedian((P[np.argmin(np.abs(t)), w_r] - P[np.argmin(np.abs(t)), s_r])[0]) > 0 else -1
    rows = []
    for k, (cj, rj, name) in enumerate(((w_c, w_r, "wrist"), (e_c, e_r, "elbow"))):
        ok = (p.conf[f, s_c] >= CONF_MIN) & (p.conf[f, cj] >= CONF_MIN)
        real = image_arm_angle_deg(xy[f, s_c], xy[f, cj], side)
        repl = image_arm_angle_deg(P[:, s_r], P[:, rj], side)
        rows.append(pd.DataFrame({"clip": c, "split": clips[c]["split"], "joint": name, "frame": f, "t_s": t,
                                  "k": np.round(t * 60).astype(int), "conf": ok, "real_deg": real,
                                  "replica_deg": repl, "err_deg": wrap(repl - real)}))
    return pd.concat(rows), (P, int(f[0])), side, xy, p


def overlay(c, vd, out, P_f0, xy, p, cm, clips, throws, side):
    rp, fps = clips[c]["release_pose_frame"], clips[c]["fps"]
    cap = cv2.VideoCapture(str(vd / c))
    s_c = COCO[throws]
    panels = []
    for off in OFFSETS_S:
        fr = int(round(rp + off * fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, fr)
        ok, img = cap.read()
        if not ok:
            continue
        P, f0 = P_f0
        i = fr - f0
        pts = xy[fr, COCO_IDX]
        # crop around the real pitcher
        cx, cy = np.nanmedian(pts[:, 0]), np.nanmedian(pts[:, 1])
        h = 3.2 * np.nanmax(np.linalg.norm(pts - [cx, cy], axis=1))
        h = max(h, 160)
        x0, y0 = int(max(cx - h / 2, 0)), int(max(cy - h / 2, 0))
        sub = img[y0:int(y0 + h), x0:int(x0 + h)].copy()
        sub = cv2.resize(sub, (360, 360))
        sc = 360 / h

        def pt(q):
            return int((q[0] - x0) * sc), int((q[1] - y0) * sc)
        # replica arm (red): shoulder-elbow-wrist (positions 0/2/4 or 1/3/5); real (green)
        r = (0, 2, 4) if throws == "L" else (1, 3, 5)
        if 0 <= i < len(P) and np.isfinite(P[i]).all():
            for a, b in ((r[0], r[1]), (r[1], r[2])):
                cv2.line(sub, pt(P[i, a]), pt(P[i, b]), (0, 0, 255), 3)
        for a, b in ((s_c[0], s_c[1]), (s_c[1], s_c[2])):
            ca, cb = p.conf[fr, a] >= CONF_MIN, p.conf[fr, b] >= CONF_MIN
            col = (0, 220, 0) if ca and cb else (0, 160, 220)   # orange-ish: low-confidence keypoint
            cv2.line(sub, pt(xy[fr, a]), pt(xy[fr, b]), col, 2)
        cv2.putText(sub, f"t={off:+.2f}s", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        panels.append(sub)
    cv2.imwrite(str(out), np.hstack(panels))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clips_dir")
    ap.add_argument("--tag", default="now")
    ap.add_argument("--overlay", nargs="*", default=[])
    ap.add_argument("--deliveries", default=str(ROOT / "web/public/data/deliveries"))
    a = ap.parse_args()
    vd, slug, dd = Path(a.clips_dir), Path(a.clips_dir).name, Path(a.deliveries)
    dl = json.loads((dd / f"{slug}.json").read_text())
    clips = json.loads((dd / f"{slug}.clips.json").read_text())
    comp = json.loads((dd / f"{slug}.compare.json").read_text())["clips"]
    rep = np.asarray(dl["joints_ft"], float)[:, SMPL_IDX]
    ri, fps_d, throws = dl["release_index"], float(dl["fps"]), dl["throws"]
    out = []
    for c, cm in comp.items():
        df, P, side, xy, p = clip_curves(c, vd, cm, clips, rep, ri, fps_d, throws)
        out.append(df)
        if c in a.overlay:
            o = ROOT / "results/m5/screens" / f"{slug.split('-')[0]}_arm2_{a.tag}_{c.replace('.mp4', '')}.png"
            overlay(c, vd, o, P, xy, p, cm, clips, throws, side)
            print("wrote", o)
    df = pd.concat(out)
    df.to_csv(ROOT / "results/m5" / slug / f"arm_curve_{a.tag}.csv", index=False)
    d = df[df["conf"]]
    pd.set_option("display.width", 200)
    for split in ("test", "train"):
        for joint in ("wrist", "elbow"):
            g = d[(d["split"] == split) & (d["joint"] == joint)]
            tab = g.groupby("k").agg(n=("clip", "nunique"), real=("real_deg", "median"), replica=("replica_deg", "median"),
                                     err_med=("err_deg", "median"), abs_err_med=("err_deg", lambda v: np.median(np.abs(v))))
            print(f"\n== {split} {joint} (deg above horizontal, throwing side; err = replica - real) ==")
            print(tab[(tab.index >= -24) & (tab.index <= 9)].round(1).to_string())
    print("\nStatcast arm_angle at release per clip, and the image angle at release:")
    for c, cm in comp.items():
        g = df[(df["clip"] == c) & (df["joint"] == "wrist") & (df["k"] == 0)]
        print(f"  {c:18s} {cm['split']:5s} statcast {cm['arm_angle_deg']}  real_img {g['real_deg'].round(1).tolist()} "
              f"replica_img {g['replica_deg'].round(1).tolist()} conf {g['conf'].tolist()}")


if __name__ == "__main__":
    main()
