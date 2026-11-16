"""M5 build: lifted clips -> one averaged, anchored delivery + accuracy metrics + viewer JSON.

usage: python scripts/m5_build.py data/video/ohtani-shohei --raw data/raw/660271_2026-....csv

Pipeline (train clips only ever feed the average; test clips are only measured):
  screen lifts -> per clip: ground on the feet, landmarks, anchor to Statcast feet -> time-warp average of
  the anchored train clips -> foot lock -> metrics (BEFORE any correction) -> bone-length-preserving
  correct_to_mean toward the season mean release -> write JSON.
Writes web/public/data/deliveries/<slug>.json + <slug>.clips.json (gitignored) and results/m5/metrics.csv.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from pitchsim.data import fetch_heights
from pitchsim.export import write_json
from pitchsim.pose.anchor import (anchor, correct_to_mean, lock_feet, mound_z_ft, release_values,
                                  segment_lengths)
from pitchsim.pose.delivery import (average_delivery, fix_leg_swaps, ground_root, has_rocker_step, jitter,
                                    landmarks, lift_snap, phase_smooth, rocker_step)
from pitchsim.pose.clips import read_labels
from pitchsim.pose.joints3d import load_lift
from pitchsim.pose.pose2d import load_pose2d
from pitchsim.pose.release import foot_strike_frame
from pitchsim.statcast import clean
from pitchsim.validation import split_by_game

ROOT = Path(__file__).resolve().parents[1]
KEYS = ["release_x_ft", "release_z_ft", "extension_ft", "arm_angle_deg"]
STAT = {"release_x_ft": "release_pos_x", "release_z_ft": "release_pos_z",
        "extension_ft": "release_extension", "arm_angle_deg": "arm_angle"}
HAND = {"R": 23, "L": 22}
HAND_TOE = {"R": (11, 10), "L": (10, 11)}  # (pivot toe, lead toe) SMPL joints for a R / L thrower

# Why 15 deg: gravity-up should be near-constant for a fixed broadcast camera.  The 20 good clips are
# within 12 deg of the median up; the one known bad lift is 56 deg off.  15 sits in that gap.
MAX_UP_TILT_DEG = 15.0
# Why 0.85: the median per-frame body bounding-box diagonal is ~1.6-1.7 m for every clip; a lift that
# collapsed (body folded / scale failure) reads far smaller.  Anything under 85% of the clip-set median is out.
MIN_EXTENT_FRAC = 0.85
PRE_COVER_FRAC = 0.85   # pre_s = longest lead-in that >= 85% of train clips cover
# 3.0 s (was 2.5): Ohtani's rocker step (lead foot steps ~1.3 ft sideways, hips sway over it) happens
# ~2.8-2.3 s before release, and the clips start ~2.9-3.0 s before release: 2.5 s cut the step off.
# 2.85 keeps 13 of 14 train clips after warping (2.9 s kept 11: clips whose step is early run out of frames).
PRE_CAP_S = 2.85
POST_S = 0.6
# Release POSE frame = hand label - 1.  The labelling rule marks the first frame the ball is visibly out of
# the hand, so at the label the hand has already whipped past release (it moves ~2 ft per frame at 60 fps):
# the 2D image-plane arm angle is 25-35 deg at label-3..-1 and jumps to ~-140 deg at the label in about
# half the clips, and the per-clip 3D arm-angle median error vs Statcast is 21.6 deg at the label, 8.0 at
# label-1, 1.8 at label-2 (controller's check, 2026-11-14).  label-1 is the last frame with the ball still
# in hand, so it is the frame whose hand position Statcast's release point describes.
RELEASE_POSE_OFFSET = -1


def boot_ci(err: pd.Series, games: pd.Series, q: float, n: int = 1000, seed: int = 0):
    """Quantile of |err| with a 95% interval from resampling held-out GAMES (clips within a game
    are correlated, so games, not clips, are the independent units)."""
    rng = np.random.default_rng(seed)
    g = games.unique()
    by = {k: err[games == k].to_numpy() for k in g}
    stats = [np.quantile(np.concatenate([by[k] for k in rng.choice(g, len(g))]), q) for _ in range(n)]
    return float(np.quantile(err, q)), *np.quantile(stats, [0.025, 0.975])


def metric_rows(name: str, truth: pd.DataFrame, pred: pd.DataFrame, games: pd.Series) -> list[dict]:
    rows = []
    for k in KEYS:
        e = (pred[k] - truth[k]).dropna()
        a = e.abs()
        g = games[a.index]
        med, mlo, mhi = boot_ci(a, g, 0.5)
        p90, plo, phi = boot_ci(a, g, 0.9)
        rows.append({"set": name, "metric": k, "n": len(a), "n_games": g.nunique(),
                     "median_abs": med, "med_lo": mlo, "med_hi": mhi,
                     "p90_abs": p90, "p90_lo": plo, "p90_hi": phi, "median_signed": float(e.median())})
    return rows


def parse_clip(c: str) -> tuple[int, int, int]:
    g, ab, pn = c.replace(".mp4", "").split("_")
    return int(g), int(ab), int(pn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clips_dir")
    ap.add_argument("--raw", required=True, help="the pitcher's Statcast CSV (data/raw/<id>_....csv)")
    ap.add_argument("--out", default=str(ROOT / "web/public/data/deliveries"))
    ap.add_argument("--results", default=None, help="default results/m5/<slug>")
    ap.add_argument("--pre-cap", type=float, default=PRE_CAP_S, help="cap on pre_s (s before release)")
    a = ap.parse_args()
    d, slug = Path(a.clips_dir), Path(a.clips_dir).name

    clips = pd.read_csv(d / "clips.csv")
    pose = pd.read_csv(d / "pose2d.csv").set_index("clip")
    labels = read_labels(slug, "release_frames")
    raw = pd.read_csv(a.raw)
    pid, throws = int(raw["pitcher"].iloc[0]), str(clips["p_throws"].iloc[0])
    h = fetch_heights([pid]).iloc[0]
    if pd.isna(h):  # anchor() scales the body to it: no height, no delivery
        raise SystemExit(f"no StatsAPI height for pitcher {pid}; cannot scale the delivery")
    height_in = int(h)
    height_ft = height_in / 12
    stat = clean(raw).set_index(["game_pk", "at_bat_number", "pitch_number"])

    # ---- yield table ------------------------------------------------------------------------
    drops: dict[str, list[str]] = {}

    def drop(reason: str, clip: str):
        drops.setdefault(reason, []).append(clip)

    for r in clips.itertuples():
        if r.status not in ("ok", "cached"):
            drop(f"download_{r.status}", f"{r.game_pk}_{r.at_bat_number}_{r.pitch_number}")
    for clip, r in pose.iterrows():
        if not r["ok"]:
            drop(f"screen_{r['reason']}", clip)

    # ---- load + screen lifts ----------------------------------------------------------------
    ok = pose[pose["ok"]]
    lift_path = lambda c: d / "lift" / c.replace(".mp4", ".npz")  # noqa: E731
    L = {c: load_lift(lift_path(c)) for c in ok.index if lift_path(c).exists()}
    for c in ok.index:
        if c not in L:
            drop("no_lift", c)
    # Screening references come from TRAIN lifts only (hold-out hygiene: test clips must not shape the
    # thresholds they are screened by); every lift, test included, is then screened against them.
    tr_lifts = [c for c in L if pose.loc[c, "split"] == "train"]
    up_med = np.median(np.stack([L[c].up for c in tr_lifts]), axis=0)
    up_med /= np.linalg.norm(up_med)
    tilt = {c: float(np.degrees(np.arccos(np.clip(l.up @ up_med, -1, 1)))) for c, l in L.items()}
    extent = {c: float(np.median(np.linalg.norm(l.joints_m.max(1) - l.joints_m.min(1), axis=1)))
              for c, l in L.items()}
    ext_med = np.median([extent[c] for c in tr_lifts])
    screened, passed = [], []
    for c in L:
        if tilt[c] > MAX_UP_TILT_DEG:
            drop("bad_up", c)
        elif extent[c] < MIN_EXTENT_FRAC * ext_med:
            drop("bad_extent", c)
        else:
            passed.append(c)
    for c in L:
        if c not in passed:
            print(f"suspect {c}: tilt {tilt[c]:.1f} deg, extent {extent[c]:.2f} m (train median {ext_med:.2f}) -> DROPPED")

    # release frame: hand label where one exists, else the detector's
    rel = {c: int(labels.loc[c, "release_frame_label"]) if c in labels.index
           else int(pose.loc[c, "release_frame"]) for c in L}
    sx = {}
    for c in L:
        if parse_clip(c) in stat.index:
            sx[c] = stat.loc[parse_clip(c)]
    for c in passed:
        if c not in sx:
            drop("no_statcast", c)
    clean_clips = [c for c in passed if c in sx]
    split = pose["split"]
    train = [c for c in clean_clips if split[c] == "train"]
    test = [c for c in clean_clips if split[c] == "test"]

    rp = {c: rel[c] + RELEASE_POSE_OFFSET for c in L}
    # Foot strike: hand label, else the 2D detector, else the train median offset before release.  Not the
    # 3D lift's (delivery.landmarks): the lift compresses the stride's depth, so its lead ankle stops ~0.45 s
    # before release while the real foot plants ~0.16 s before (hand labels); planting there made the body
    # pause on a planted leg.  2D pose and lift share frame indices (m5_lift lifts the pose2d frames).
    fs_lab = read_labels(slug, "foot_strike_frames").get("foot_strike_frame_label", pd.Series(dtype=float))
    fs, fs_src = {}, {}
    for c in L:
        f = int(fs_lab[c]) if c in fs_lab.index else foot_strike_frame(
            load_pose2d(d / "pose2d" / c.replace(".mp4", ".npz")), throws, rp[c])
        if f is not None:
            fs[c], fs_src[c] = f, "label" if c in fs_lab.index else "2d"
    fs_off_s = float(np.median([(rp[c] - fs[c]) / L[c].fps for c in fs if split[c] == "train"]))
    for c in L:
        if c not in fs:
            fs[c], fs_src[c] = rp[c] - int(round(fs_off_s * L[c].fps)), "median"
    # Canonicalise EACH clip before averaging: the clips come from different games and camera set-ups, so
    # their lift (camera) frames differ; averaging there would mix frames.  The lifts' root translation is
    # unusable (pelvis depth jumps by metres), so each clip is first grounded on its feet
    # (delivery.ground_root; foot strike is found on the pivot-grounded version), then anchored into
    # Statcast ft with its own gravity-up, the StatsAPI height and the pivot toe on the rubber.
    # Lift repairs first (delivery.fix_leg_swaps / lift_snap): left/right leg swaps are undone, and a lift
    # that sits frozen and then snaps is only used after its snap.  Both were in the 5 train clips with no
    # rocker step, and the snaps sat at the leg lift: averaged in, they were the leg-lift hitch.
    J, snap, n_swapped = {}, {}, {}
    for c in L:
        J[c], n_swapped[c] = fix_leg_swaps(L[c].joints_m, rp[c])
        snap[c] = lift_snap(J[c], rp[c], L[c].fps)
    A, marks, rock = {}, {}, {}
    for c in L:
        # leg lift from the 3D lift (Ryan likes the leg lift); foot strike from 2D / labels
        ll_c = landmarks(ground_root(J[c], throws), rp[c], throws, up=L[c].up)["leg_lift"]
        A[c] = anchor(ground_root(J[c], throws, fs[c]), L[c].up, height_ft, throws, rp[c], fps=L[c].fps)
        marks[c] = {"leg_lift": ll_c, "foot_strike": fs[c], "release": rp[c], "fps": L[c].fps}
        # a step or leg-lift peak found within 0.1 s after a snap is the snap itself, not the motion
        after_snap = lambda f: snap[c] is not None and f < snap[c] + int(round(0.1 * L[c].fps))  # noqa: E731
        rock[c] = None if after_snap(ll_c - int(round(0.8 * L[c].fps))) else rocker_step(A[c], throws, ll_c, L[c].fps)
        if after_snap(ll_c):
            marks[c]["leg_lift"] = None
    # leg lift lost in a snap: stand-in at the median offset before release of the clips that have one
    ll_off_s = float(np.median([(rp[c] - marks[c]["leg_lift"]) / L[c].fps for c in train
                                if marks[c]["leg_lift"] is not None]))
    for c in L:
        if marks[c]["leg_lift"] is None:
            marks[c]["leg_lift"] = rp[c] - int(round(ll_off_s * L[c].fps))
    # Rocker-step knot: lines the step up across clips.  A clip whose lift shows no step (the lift or the
    # tracker missed it) gets a stand-in knot at the median step -> leg-lift gap, so its warp stays ~slope 1,
    # and sits out the average until 0.9 s before the template leg lift, easing in (smootherstep) over the
    # next 0.4 s, well clear of the leg lift: averaging it in would shrink the step, and a sudden change of
    # the clip mix is itself a jump.
    # A pitcher with no rocker step (most of his train clips show none): no knot and no fade; a stray
    # clip's "step" is lift drift, not his motion (delivery.has_rocker_step).
    stepper = has_rocker_step([rock[c] for c in train])
    if not stepper:
        rock = {c: None for c in L}
    else:
        gap_s = float(np.median([(marks[c]["leg_lift"] - rock[c]) / L[c].fps for c in train if rock[c] is not None]))
        for c in L:
            marks[c]["rocker"] = rock[c] if rock[c] is not None else marks[c]["leg_lift"] - int(round(gap_s * L[c].fps))
    tmpl_ll_s = float(np.median([(marks[c]["leg_lift"] - rp[c]) / L[c].fps for c in train]))
    def fade(tau):
        u = np.clip((tau - (tmpl_ll_s - 0.9)) / 0.4, 0.0, 1.0)
        return u ** 3 * (u * (6 * u - 15) + 10)
    avail = np.array([rp[c] / L[c].fps for c in train])
    pre_s = float(np.floor(min(a.pre_cap, np.quantile(avail, 1 - PRE_COVER_FRAC, method="lower")) * 100) / 100)
    excl: list = []
    avg_raw, ri = average_delivery([A[c] for c in train], [marks[c] for c in train], pre_s,
                                   fps_out=60.0, post_s=POST_S, excluded=excl,
                                   weights=[None if not stepper or rock[c] is not None or snap[c] is not None else fade
                                            for c in train],
                                   valid_from=[snap[c] for c in train])
    short = list(dict.fromkeys(train[i] for i, _ in excl))
    for i, why in excl:
        drop(why, train[i])
    train_used = [c for c in train if c not in short]
    # Averaging lets planted feet creep a little (clips disagree slightly); lock them again on the average.
    # leg lift / foot strike / rocker step of the template: median offsets before release (s), in output
    # frames (the same medians average_delivery warps to)
    ll_s, fs_s, rk_s = (np.median([(marks[c][k] - marks[c]["release"]) / L[c].fps for c in train_used])
                        if k in marks[train_used[0]] else None for k in ("leg_lift", "foot_strike", "rocker"))
    ll, fsi, rki = (None if v is None else int(round(ri + v * 60.0)) for v in (ll_s, fs_s, rk_s))
    # Extra smoothing before the leg-lift peak only (delivery.phase_smooth): the windup is slow, so the
    # averaged lifts' leftover noise read as a wobble there; the stride and throw keep today's smoothing.
    avg_raw = phase_smooth(avg_raw, 60.0, slow_until=ll)
    avg_ft = lock_feet(avg_raw, 60.0, v_lo_ftps=1.0, v_hi_ftps=2.5)
    rv_avg = release_values(avg_ft, ri, throws)

    # ---- metrics BEFORE correct_to_mean (hold-out: test clips never fed the average) ---------
    def truth_of(cs):
        return pd.DataFrame({k: [sx[c][STAT[k]] for c in cs] for k in KEYS}, index=cs)

    def own_anchor(c):
        return A[c], release_values(A[c], rp[c], throws)

    games = pd.Series({c: parse_clip(c)[0] for c in L})
    own = pd.DataFrame([own_anchor(c)[1] for c in test], index=test)
    avg_for_test = pd.DataFrame([rv_avg] * len(test), index=test)
    mrows = (metric_rows("own_lift_test", truth_of(test), own, games[test])
             + metric_rows("averaged_vs_test", truth_of(test), avg_for_test, games[test]))
    # Sensitivity: test clips dropped by screening, scored anyway so screening cannot flatter the numbers.
    dropped_test = [c for c in L if split[c] == "test" and c in sx and c not in test]
    if dropped_test:
        every = test + dropped_test
        own_all = pd.DataFrame([own_anchor(c)[1] for c in every], index=every)
        mrows += metric_rows("own_lift_test_incl_screened", truth_of(every), own_all, games[every])
    metrics = pd.DataFrame(mrows)

    # ---- jitter: per-clip anchored (same window around release) vs the final delivery ----------
    raw_jit = []
    for c in train_used:
        j, _ = own_anchor(c)
        f = L[c].fps
        raw_jit.append(jitter(j[max(rp[c] - int(pre_s * f), 0): rp[c] + int(POST_S * f) + 1], f))

    # ---- correct_to_mean toward the season mean of bases-empty release values (Statcast, 2026) ----
    # Target = mean over TRAIN games only (same split_by_game seed as the clip split), so the test clips'
    # pitches never pull the delivery that the 2D test score is computed on (hold-out hygiene).
    cl = split_by_game(clean(raw), seed=0)[0]
    bases_empty = cl[cl[["on_1b", "on_2b", "on_3b"]].isna().all(axis=1)]
    target = {k: float(bases_empty[STAT[k]].mean()) for k in KEYS}
    cinfo: dict = {}
    final = correct_to_mean(avg_ft, ri, fsi, throws, target, leg_lift_index=ll, info=cinfo)
    rv_fin = release_values(final, ri, throws)
    miss = {k: rv_fin[k] - target[k] for k in KEYS}
    if (max(abs(miss[k]) for k in KEYS[:3]) > 0.05 or abs(miss["arm_angle_deg"]) > 1.0
            or not cinfo.get("converged", False)):
        raise SystemExit(f"correct_to_mean did not converge to 0.05 ft / 1 deg: misses {miss}, info {cinfo}")
    # What the correction did to the body: bone lengths (vs its input; and the foot lock vs the raw
    # average), and the planted feet (pivot toe until foot strike, lead toe after) in the final delivery.
    seg_corr_ft = float(np.abs(segment_lengths(final) - segment_lengths(avg_ft)).max())
    seg_lock_ft = float(np.abs(segment_lengths(avg_ft) - segment_lengths(avg_raw)).max())
    seg_cv = segment_lengths(final).std(0) / segment_lengths(final).mean(0)
    piv, lead = HAND_TOE[throws]
    feet = {}
    for nm, k, fr in (("pivot", piv, slice(0, fsi + 1)), ("lead", lead, slice(fsi, None))):
        p = final[fr, k]
        feet[nm] = {"drift_ft": float(np.ptp(p[:, :2], axis=0).max()),
                    "height_err_ft": float(np.abs(p[:, 2] - mound_z_ft(p[:, 0], p[:, 1])).max())}
    stride_ft = float(np.linalg.norm(final[fsi, lead, :2] - final[0, piv, :2]))
    # Rocker step size: the lead toe's horizontal offset from frame 0 at leg lift - 0.7 s (on the step's
    # plateau), final delivery vs the anchored clips that show a step (before averaging)
    def step_ft(j, l, f):
        return float(np.linalg.norm(j[l - int(round(0.7 * f)), lead, :2] - j[0, lead, :2]))
    rocker = {"has_step": stepper, "final_step_ft": step_ft(final, ll, 60.0),
              "final_step_s_before_release": None if rki is None else (ri - rki) / 60.0,
              "clips_step_ft_median": float(np.median([step_ft(A[c], marks[c]["leg_lift"], L[c].fps)
                                                       for c in train_used if rock[c] is not None])) if stepper else None,
              "clips_step_s_before_release_median": None if rk_s is None else -rk_s,
              "clips_with_step": [c for c in train_used if rock[c] is not None],
              "clips_without_step": [c for c in train_used if rock[c] is None]}
    repairs = {c: {"legs_swapped_frames": n_swapped[c], "snap_s_before_release": None if snap[c] is None
                   else round((rp[c] - snap[c]) / L[c].fps, 3)} for c in L if n_swapped[c] or snap[c] is not None}

    # ---- release-frame detector error vs hand labels (in-sample unless fresh) ------------------
    lab = labels.loc[[c for c in labels.index if c in pose.index]].copy()
    lab["err"] = pose.loc[lab.index, "release_frame"] - lab["release_frame_label"]

    # ---- outputs ---------------------------------------------------------------------------
    out = Path(a.out)
    write_json({"schema": 1, "pitcher_id": pid, "slug": slug, "throws": throws, "height_in": height_in,
                "fps": 60.0, "release_index": ri, "pre_s": pre_s, "post_s": POST_S,
                "joints_ft": np.round(final, 3).tolist(), "n_clips": len(train_used),
                "lifter": str(next(iter(L.values())).lifter)}, out / f"{slug}.json")
    screen_of = {c: r for r, cs in drops.items() for c in cs}  # its drop reason (one per clip), else "ok"
    info = {}
    for c in L:
        j, rv = own_anchor(c)
        info[c] = {"release_frame": rel[c], "release_pose_frame": rp[c], "fps": float(L[c].fps),
                   "release": [round(float(v), 3) for v in j[rp[c], HAND[throws]]],
                   "extension_ft": round(rv["extension_ft"], 3),
                   "arm_angle_deg": round(rv["arm_angle_deg"], 2), "split": str(split[c]),
                   "screen": screen_of.get(c, "ok")}
    write_json(info, out / f"{slug}.clips.json")
    res = Path(a.results) if a.results else ROOT / "results/m5" / slug  # per pitcher: builds don't overwrite
    res.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(res / "metrics.csv", index=False)
    write_json({"target": target, "before": rv_avg, "after": rv_fin, "applied": cinfo,
                "stride_ft": stride_ft, "max_segment_change_correction_ft": seg_corr_ft,
                "max_segment_change_foot_lock_ft": seg_lock_ft, "planted_feet": feet,
                "leg_lift_index": ll, "foot_strike_index": fsi, "rocker_index": rki, "release_index": ri,
                "foot_strike_to_release_s": (ri - fsi) / 60.0, "rocker": rocker,
                "foot_strike_source": fs_src, "lift_repairs": repairs}, res / "correction.json")

    pd.set_option("display.width", 220)
    print(f"\nheight {height_in} in; pre_s {pre_s}; release_index {ri}; rocker_index {rki}; leg_lift_index {ll}; foot_strike_index {fsi} "
          f"(foot strike -> release {(ri - fsi) / 60.0:.3f} s; sources {pd.Series(fs_src).value_counts().to_dict()}, "
          f"median fallback {fs_off_s:.3f} s); "
          f"train used {len(train_used)}, test {len(test)}")
    print("\nYIELD  downloaded:", int(clips["status"].isin(["ok", "cached"]).sum()), "of", len(clips))
    for k, v in drops.items():
        print(f"  dropped {k}: {len(v)} {v if len(v) < 6 else ''}")
    print(f"  used: train {len(train_used)}, test {len(test)}")
    print("\nMETRICS (ft / deg), before correct_to_mean\n", metrics.round(3).to_string(index=False))
    print(f"\nJITTER (jerk RMS, ft/s^3): per-clip median {np.median(raw_jit):.0f} "
          f"(IQR {np.percentile(raw_jit, 25):.0f}-{np.percentile(raw_jit, 75):.0f})  "
          f"averaged+anchored {jitter(avg_ft, 60.0):.0f}  final {jitter(final, 60.0):.0f}")
    print("\ncorrect_to_mean   before -> after (target)")
    for k in KEYS:
        print(f"  {k}: {rv_avg[k]:.3f} -> {rv_fin[k]:.3f} ({target[k]:.3f})  gap {target[k] - rv_avg[k]:+.3f}")
    print("  applied:", {k: np.round(v, 3).tolist() if isinstance(v, (list, float)) else v for k, v in cinfo.items()})
    print(f"  stride (pivot toe -> planted lead toe, horizontal) {stride_ft:.2f} ft: CONSTRUCTED to match "
          f"Statcast's mean extension; this camera cannot measure it")
    print(f"  max bone-length change: correction {seg_corr_ft * 304.8:.4f} mm, foot lock {seg_lock_ft * 304.8:.1f} mm; "
          f"bone-length variation over time in the final (CV, from averaging): median {np.median(seg_cv):.3f}, "
          f"max {seg_cv.max():.3f}")
    for nm, v in feet.items():
        print(f"  planted {nm} toe: drift {v['drift_ft'] * 12:.4f} in, height error {v['height_err_ft'] * 12:.4f} in")
    ext = {c.replace(".mp4", ""): round(own_anchor(c)[1]["extension_ft"], 2) for c in clean_clips}
    print("  lift repairs (leg swaps undone; snap = clip used only after it):", repairs)
    if stepper:
        print(f"  rocker step: final {rocker['final_step_ft']:.2f} ft, knot {rocker['final_step_s_before_release']:.2f} s "
              f"before release; clips' median {rocker['clips_step_ft_median']:.2f} ft; "
              f"{len(rocker['clips_with_step'])} train clips with a step, without: {rocker['clips_without_step']}")
    else:
        print(f"  rocker step: none (most train clips show no step); lead toe moved {rocker['final_step_ft']:.2f} ft "
              f"by leg lift - 0.7 s in the final")
    print("\nper-clip extension BEFORE correction (ft, label-1):", ext)
    for nm, g in (("IN-SAMPLE", lab[lab["fresh"] == "no"]), ("FRESH", lab[lab["fresh"] == "yes"])):
        print(f"\nrelease detector vs labels {nm}: n={len(g)} median|err|={g['err'].abs().median()} "
              f"max|err|={g['err'].abs().max()} within2={(g['err'].abs() <= 2).sum()}/{len(g)}")


if __name__ == "__main__":
    main()
