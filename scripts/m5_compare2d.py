"""M5 Task 8: how closely does the replica delivery match real clips, in the broadcast image?

usage: python scripts/m5_compare2d.py [data/video/ohtani-shohei]

For each screened-ok clip, the replica's joints at the clip's times (t = (frame - release_pose_frame) / fps,
over the delivery window) are projected into the clip with ONE fitted affine camera + a per-frame pan
(pitchsim.pose.compare2d) and compared with the clip's RTMPose keypoints.  Error unit: torso lengths.

The replica is the delivery JSON's body WITHOUT the viewer's per-pitch hand pin (the viewer blends the
throwing arm onto that pitch's Statcast release over +-0.15 s; the rest of the body is the delivery's).
Unpinned is the honest number: pinning moves the hand to where the pitch was released, by construction.

Context rows, all fit and scored the same way:
  naive_static : the delivery's release frame held for every t (what the viewer showed before M5)
  own_lift     : the clip's own 3D lift (leg swaps undone) -- the best a lifted 3D body does on its own 2D
  floor_2d     : another real delivery's 2D keypoints (train clips from OTHER games, aligned at release),
                 padded z = 0 so the same fit is a 2D affine + pan -- how different his deliveries are
  floor_lift   : another real delivery's 3D lift, same 3D fit as the replica -- the same comparison but with
                 the lifter's SMPL-vs-COCO joint mismatch the replica also carries
"Indistinguishable" ~= replica close to the floor.  Train clips fed the delivery (in-sample); TEST clips are
the headline.  Nothing here is tuned: the window, threshold and phases come from the build or are fixed.

Writes results/m5/compare2d.csv (summary), results/m5/compare2d_frames.csv.gz (every error), and
web/public/data/deliveries/<slug>.compare.json (fitted cameras + per-clip scores, numbers only) for the page.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pitchsim.export import write_json
from pitchsim.pose.compare2d import (CONF_MIN, COCO_IDX, GROUPS, JOINT_NAMES, PHASES, SMPL_IDX,
                                     TORSO_FRAC_OF_HEIGHT, boot_median_ci, phase_index, sample_frames,
                                     score_sequence, shot_start, torso_px)
from pitchsim.pose.delivery import fix_leg_swaps
from pitchsim.pose.joints3d import load_lift
from pitchsim.pose.pose2d import load_pose2d
from pitchsim.statcast import clean

ROOT = Path(__file__).resolve().parents[1]
LEAD_KNEE = {"R": 4, "L": 5}  # SMPL: the glove-side knee lifts
# Leg-lift start: the lead knee has risen 10% of the way from its set height to its peak.  Why 10%: the
# knee sits still within ~0.05 ft before the lift, then rises ~3 ft; 10% is clear of the noise, early in the rise.
LIFT_START_FRAC = 0.10


def phase_bounds_s(joints: np.ndarray, throws: str, ri: int, corr: dict) -> list[float]:
    """Phase boundaries (s from release) from the delivery's landmarks (m5_build's correction.json):
    set/rocker | leg-lift start | leg-lift peak | foot strike | release (t = 0)."""
    rk, ll, fs = corr["rocker_index"], corr["leg_lift_index"], corr["foot_strike_index"]
    kz = joints[:, LEAD_KNEE[throws], 2]
    base = kz[rk:ll].min()
    above = np.nonzero(kz[rk:ll + 1] > base + LIFT_START_FRAC * (kz[ll] - base))[0]
    ls = rk + int(above[0])
    return [(ls - ri) / 60.0, (ll - ri) / 60.0, (fs - ri) / 60.0, 0.0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clips_dir", nargs="?", default=str(ROOT / "data/video/ohtani-shohei"))
    ap.add_argument("--deliveries", default=str(ROOT / "web/public/data/deliveries"))
    ap.add_argument("--results", default=str(ROOT / "results/m5"))
    ap.add_argument("--raw", default=str(ROOT / "data/raw/660271_2026-03-01_2026-11-30.csv"))
    a = ap.parse_args()
    vd, slug, dd, res = Path(a.clips_dir), Path(a.clips_dir).name, Path(a.deliveries), Path(a.results)

    dl = json.loads((dd / f"{slug}.json").read_text())
    clips = json.loads((dd / f"{slug}.clips.json").read_text())
    corr = json.loads((res / "correction.json").read_text())
    joints = np.asarray(dl["joints_ft"], float)
    ri, fps_d, throws = dl["release_index"], float(dl["fps"]), dl["throws"]
    assert corr["release_index"] == ri, "correction.json is from another build; re-run m5_build.py"
    rep = joints[:, SMPL_IDX]
    bounds = phase_bounds_s(joints, throws, ri, corr)
    torso_in = TORSO_FRAC_OF_HEIGHT * dl["height_in"]

    ok = [c for c, v in clips.items() if v["screen"] == "ok"]
    game = {c: int(c.split("_")[0]) for c in ok}
    P, X2, W, Lf, T, Fr, TOR, CUT = {}, {}, {}, {}, {}, {}, {}, {}
    for c in ok:
        p = load_pose2d(vd / "pose2d" / c.replace(".mp4", ".npz"))
        rp, fps = clips[c]["release_pose_frame"], clips[c]["fps"]
        f = np.arange(len(p.xy))
        t = (f - rp) / fps
        X2[c] = p.xy[:, COCO_IDX].astype(float)
        W[c] = p.conf[:, COCO_IDX] >= CONF_MIN
        # Some clips open on another shot (a batter close-up) and cut to the pitcher mid-windup: before the
        # cut neither the 2D track nor the lift is the pitcher, and one camera cannot fit two shots.  Those
        # frames are dropped everywhere (scored clip and reference alike).
        CUT[c] = shot_start(X2[c], W[c], rp, fps)
        W[c][:CUT[c]] = False
        # the delivery window (the replica exists there), from the shot with the release in it
        keep = (t >= -dl["pre_s"]) & (t <= dl["post_s"]) & (f >= CUT[c])
        Fr[c], T[c] = f[keep], t[keep]
        TOR[c] = torso_px(X2[c][keep], W[c][keep])
        Lf[c] = fix_leg_swaps(load_lift(vd / "lift" / c.replace(".mp4", ".npz")).joints_m, rp)[0][:, SMPL_IDX]
        Lf[c][:CUT[c]] = np.nan
        P[c] = (rp, fps)

    def at_clip_times(seq, d, t):
        """Reference d's sequence (per its own frames) sampled at times t s from d's release."""
        rp, fps = P[d]
        return sample_frames(seq, rp + t * fps)

    train = [c for c in ok if clips[c]["split"] == "train"]
    recs, cams = [], {}
    for c in ok:
        f, t = Fr[c], T[c]
        x, w = X2[c][f], W[c][f]
        cands = {"replica": sample_frames(rep, ri + t * fps_d),
                 "naive_static": np.repeat(rep[ri][None], len(t), 0),
                 "own_lift": Lf[c][f]}
        for d in train:
            if game[d] == game[c]:
                continue  # same-game deliveries share a camera and a day: a cross-game floor is the fair one
            x2 = np.where(W[d][..., None], X2[d], np.nan)  # unconfident reference points drop out of the fit
            cands[f"floor_2d|{d}"] = np.concatenate([at_clip_times(x2, d, t), np.zeros((len(t), 12, 1))], -1)
            cands[f"floor_lift|{d}"] = at_clip_times(Lf[d], d, t)
        ph = phase_index(t, bounds)
        for name, X in cands.items():
            err, (A, b, okf) = score_sequence(X, x, w, TOR[c])
            if name == "replica":
                cams[c] = {"A": A, "b": b, "ok": okf, "err": err}
            m = np.isfinite(err)
            fi, ji = np.nonzero(m)
            method, _, ref = name.partition("|")
            recs.append(pd.DataFrame({"clip": c, "game": game[c], "split": clips[c]["split"], "method": method,
                                      "ref": ref, "t_s": t[fi], "phase": np.array(PHASES)[ph[fi]],
                                      "joint": np.array(JOINT_NAMES)[ji], "err": err[m]}))
    df = pd.concat(recs, ignore_index=True)
    grp_of = {JOINT_NAMES[i]: g for g, ix in GROUPS.items() for i in ix}
    df["group"] = df["joint"].map(grp_of)

    # ---- summary -----------------------------------------------------------------------------------
    rows = []
    for (split, method), g in df.groupby(["split", "method"]):
        for gname, gsel in [("all", g)] + [(k, g[g["group"] == k]) for k in GROUPS]:
            for pname, s in [("all", gsel)] + [(k, gsel[gsel["phase"] == k]) for k in PHASES]:
                if s.empty:
                    continue
                row = {"split": split, "method": method, "group": gname, "phase": pname,
                       "n_clips": s["clip"].nunique(), "n_games": s["game"].nunique(), "n_points": len(s),
                       "median": s["err"].median(), "p90": s["err"].quantile(0.9)}
                if gname == "all" and pname == "all":
                    by = s.groupby("clip")
                    row["med_lo"], row["med_hi"] = boot_median_ci(
                        [v.to_numpy() for _, v in by["err"]], [game[k] for k in by.groups])[1:]
                rows.append(row)
    summ = pd.DataFrame(rows)
    summ["median_in"], summ["p90_in"] = summ["median"] * torso_in, summ["p90"] * torso_in
    res.mkdir(parents=True, exist_ok=True)
    summ.to_csv(res / "compare2d.csv", index=False)
    df.to_csv(res / "compare2d_frames.csv.gz", index=False)

    # ---- export for the page: the replica camera per clip + its scores (numbers only, no video) ----
    def clip_score(c, method):
        s = df[(df["clip"] == c) & (df["method"] == method)]
        return {"overall": round(float(s["err"].median()), 4),
                "phases": {k: round(float(v.median()), 4) for k, v in s.groupby("phase")["err"]}}

    out = {"schema": 1, "slug": slug, "conf_min": CONF_MIN, "torso_in": round(torso_in, 2),
           "phases": [{"name": n, "t0_s": lo, "t1_s": hi} for n, lo, hi in
                      zip(PHASES, [-dl["pre_s"]] + bounds, bounds + [dl["post_s"]])],
           "smpl_idx": SMPL_IDX.tolist(), "joint_names": JOINT_NAMES,
           "note": "Camera: image_px = A @ joint_ft (Statcast ft) + b[frame - frame0]. Score: median 2D error in "
                   "torso lengths (replica unpinned). floor_2d: other real deliveries' 2D, same fit.",
           "clips": {}}
    # Each clip's Statcast release + arm angle: the page pins the replica's hand there, as the viewer does.
    stat = clean(pd.read_csv(a.raw)).set_index(["game_pk", "at_bat_number", "pitch_number"])
    for c, cm in cams.items():
        f = Fr[c]
        r = stat.loc[tuple(int(v) for v in c[:-4].split("_"))]
        out["clips"][c] = {"split": clips[c]["split"], "fps": P[c][1], "release_pose_frame": P[c][0],
                           "frame0": int(f[0]), "shot_start_frame": CUT[c], "torso_px": round(TOR[c], 2),
                           "A": np.round(cm["A"], 5).tolist(),
                           "release_ft": [round(float(r[k]), 3) for k in ("release_pos_x", "release_pos_y", "release_pos_z")],
                           "arm_angle_deg": None if pd.isna(r.get("arm_angle")) else round(float(r["arm_angle"]), 1),
                           "b": [None if not o else np.round(v, 2).tolist() for v, o in zip(cm["b"], cm["ok"])],
                           "score": clip_score(c, "replica"), "naive_static": clip_score(c, "naive_static"),
                           "floor_2d": clip_score(c, "floor_2d"), "own_lift": clip_score(c, "own_lift")}
    write_json(out, dd / f"{slug}.compare.json")

    # ---- print ----------------------------------------------------------------------------------------
    pd.set_option("display.width", 220)
    print(f"phase bounds (s from release): {np.round(bounds, 3).tolist()}  torso ~{torso_in:.1f} in "
          f"(Winter 0.288 x {dl['height_in']} in); conf >= {CONF_MIN}; clips ok: "
          f"train {len(train)}, test {len(ok) - len(train)}")
    for c in ok:
        if CUT[c] > Fr[c][0] - 1 and CUT[c] > 0:
            print(f"  camera cut: {c} ({clips[c]['split']}) scored from {(CUT[c] - P[c][0]) / P[c][1]:+.2f} s "
                  f"(window starts {-dl['pre_s']:+.2f} s)")
    order = ["replica", "floor_2d", "floor_lift", "own_lift", "naive_static"]
    for split in ("test", "train"):
        s = summ[(summ["split"] == split)].set_index(["method", "group", "phase"])
        print(f"\n== {split.upper()} clips: median (p90) error, torso lengths [~inches] ==")
        tab = pd.DataFrame({m: {f"{gname}/{pname}": (f"{s.loc[(m, gname, pname), 'median']:.3f} "
                                                     f"({s.loc[(m, gname, pname), 'p90']:.3f}) "
                                                     f"[{s.loc[(m, gname, pname), 'median_in']:.1f}]")
                                for gname in ["all", *GROUPS] for pname in ["all", *PHASES]
                                if (m, gname, pname) in s.index}
                            for m in order})
        print(tab.to_string())
        ci = s.loc[[(m, "all", "all") for m in order], ["med_lo", "med_hi"]].round(3)
        print("95% CI of the overall median (games resampled):\n", ci.to_string())
    print("\nper-clip replica overall median (torso lengths) vs its floor_2d / naive:")
    for c, v in out["clips"].items():
        print(f"  {c:18s} {v['split']:5s} {v['score']['overall']:.3f}  floor {v['floor_2d']['overall']:.3f}  "
              f"naive {v['naive_static']['overall']:.3f}  own_lift {v['own_lift']['overall']:.3f}")


if __name__ == "__main__":
    main()
