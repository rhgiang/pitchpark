"""M5 build: lifted clips -> one averaged, anchored delivery + accuracy metrics + viewer JSON.

usage: python scripts/m5_build.py data/video/ohtani-shohei [--raw data/raw/660271_2026-....csv]

Pipeline (train clips only ever feed the average; test clips are only measured):
  screen lifts -> landmarks -> time-warp average -> anchor to Statcast feet -> metrics (BEFORE any
  correction) -> correct_to_mean toward the season mean release -> write JSON.
Writes web/public/data/deliveries/<slug>.json + <slug>.clips.json (gitignored) and results/m5/metrics.csv.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from pitchsim.data import fetch_heights
from pitchsim.export import write_json
from pitchsim.pose.anchor import anchor, correct_to_mean, release_values
from pitchsim.pose.delivery import average_delivery, ground_root, jitter, landmarks
from pitchsim.pose.joints3d import load_lift
from pitchsim.statcast import clean

ROOT = Path(__file__).resolve().parents[1]
LABELS = ROOT / "src/pitchsim/pose/labels/ohtani_release_frames.csv"
KEYS = ["release_x_ft", "release_z_ft", "extension_ft", "arm_angle_deg"]
STAT = {"release_x_ft": "release_pos_x", "release_z_ft": "release_pos_z",
        "extension_ft": "release_extension", "arm_angle_deg": "arm_angle"}
HAND = {"R": 23, "L": 22}

# Why 15 deg: gravity-up should be near-constant for a fixed broadcast camera.  The 20 good clips are
# within 12 deg of the median up; the one known bad lift is 56 deg off.  15 sits in that gap.
MAX_UP_TILT_DEG = 15.0
# Why 0.85: the median per-frame body bounding-box diagonal is ~1.6-1.7 m for every clip; a lift that
# collapsed (body folded / scale failure) reads far smaller.  Anything under 85% of the clip-set median is out.
MIN_EXTENT_FRAC = 0.85
PRE_COVER_FRAC = 0.85   # pre_s = longest lead-in that >= 85% of train clips cover
PRE_CAP_S = 2.5
POST_S = 0.6


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
    ap.add_argument("--raw", default=str(ROOT / "data/raw/660271_2026-03-01_2026-11-30.csv"))
    ap.add_argument("--out", default=str(ROOT / "web/public/data/deliveries"))
    ap.add_argument("--results", default=str(ROOT / "results/m5"))
    a = ap.parse_args()
    d, slug = Path(a.clips_dir), Path(a.clips_dir).name

    clips = pd.read_csv(d / "clips.csv")
    pose = pd.read_csv(d / "pose2d.csv").set_index("clip")
    labels = pd.read_csv(LABELS).set_index("clip")
    raw = pd.read_csv(a.raw)
    pid, throws = int(raw["pitcher"].iloc[0]), str(clips["p_throws"].iloc[0])
    height_in = int(fetch_heights([pid]).iloc[0])
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
    up_med = np.median(np.stack([l.up for l in L.values()]), axis=0)
    up_med /= np.linalg.norm(up_med)
    tilt = {c: float(np.degrees(np.arccos(np.clip(l.up @ up_med, -1, 1)))) for c, l in L.items()}
    extent = {c: float(np.median(np.linalg.norm(l.joints_m.max(1) - l.joints_m.min(1), axis=1)))
              for c, l in L.items()}
    ext_med = np.median(list(extent.values()))
    screened, passed = [], []
    for c in L:
        if tilt[c] > MAX_UP_TILT_DEG:
            drop("bad_up", c)
        elif extent[c] < MIN_EXTENT_FRAC * ext_med:
            drop("bad_extent", c)
        else:
            passed.append(c)
    suspect = "823691_49_6.mp4"
    print(f"suspect {suspect}: tilt {tilt[suspect]:.1f} deg, extent {extent[suspect]:.2f} m "
          f"(set median {ext_med:.2f}) -> {'kept' if suspect in passed else 'DROPPED'}")

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

    # The lifts' root translation is unusable (pelvis depth jumps by metres), so ground each clip on its
    # feet first (delivery.ground_root).  Foot strike itself is found on the pivot-grounded version.
    G, marks = {}, {}
    for c in L:
        lm = landmarks(ground_root(L[c].joints_m, throws), rel[c], throws, up=L[c].up)
        G[c] = ground_root(L[c].joints_m, throws, lm["foot_strike"])
        marks[c] = {**lm, "fps": L[c].fps}
    avail = np.array([rel[c] / L[c].fps for c in train])
    pre_s = float(np.floor(min(PRE_CAP_S, np.quantile(avail, 1 - PRE_COVER_FRAC, method="lower")) * 100) / 100)
    excl: list = []
    avg_m, ri = average_delivery([G[c] for c in train], [marks[c] for c in train], pre_s,
                                 fps_out=60.0, post_s=POST_S, excluded=excl)
    short = list(dict.fromkeys(train[i] for i, _ in excl))
    for c in short:
        drop("short", c)
    train_used = [c for c in train if c not in short]
    # average_delivery averages metres in the lift frame; anchor needs one gravity-up for all of them.
    # The camera is fixed, so use the train-median up (per-clip tilt was screened above).
    up_train = np.median(np.stack([L[c].up for c in train_used]), axis=0)
    up_train /= np.linalg.norm(up_train)
    avg_ft = anchor(avg_m, up_train, height_ft, throws, ri)
    rv_avg = release_values(avg_ft, ri, throws)

    # ---- metrics BEFORE correct_to_mean (hold-out: test clips never fed the average) ---------
    def truth_of(cs):
        return pd.DataFrame({k: [sx[c][STAT[k]] for c in cs] for k in KEYS}, index=cs)

    def own_anchor(c):
        j = anchor(G[c], L[c].up, height_ft, throws, rel[c], fps=L[c].fps)
        return j, release_values(j, rel[c], throws)

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
        raw_jit.append(jitter(j[max(rel[c] - int(pre_s * f), 0): rel[c] + int(POST_S * f) + 1], f))

    # ---- correct_to_mean toward the season mean of bases-empty release values (Statcast, 2026) ----
    cl = clean(raw)
    bases_empty = cl[cl[["on_1b", "on_2b", "on_3b"]].isna().all(axis=1)]
    target = {k: float(bases_empty[STAT[k]].mean()) for k in KEYS}
    # foot strike of the template: median offset before release (s), in output frames
    fs_s = np.median([(marks[c]["foot_strike"] - marks[c]["release"]) / L[c].fps for c in train_used])
    fs = int(round(ri + fs_s * 60.0))
    final = correct_to_mean(avg_ft, ri, fs, throws, target)
    rv_fin = release_values(final, ri, throws)
    # The z shift moves the whole body (feet included); > 0.15 ft would float/sink the feet on the mound.
    dz = target["release_z_ft"] - rv_avg["release_z_ft"]

    # ---- release-frame detector error vs hand labels (in-sample unless fresh) ------------------
    lab = labels.loc[[c for c in labels.index if c in pose.index]].copy()
    lab["err"] = pose.loc[lab.index, "release_frame"] - lab["release_frame_label"]

    # ---- outputs ---------------------------------------------------------------------------
    out = Path(a.out)
    write_json({"schema": 1, "pitcher_id": pid, "slug": slug, "throws": throws, "height_in": height_in,
                "fps": 60.0, "release_index": ri, "pre_s": pre_s, "post_s": POST_S,
                "joints_ft": np.round(final, 3).tolist(), "n_clips": len(train_used),
                "lifter": str(next(iter(L.values())).lifter)}, out / f"{slug}.json")
    info = {}
    for c in L:
        j, rv = own_anchor(c)
        info[c] = {"release_frame": rel[c], "fps": float(L[c].fps),
                   "release": [round(float(v), 3) for v in j[rel[c], HAND[throws]]],
                   "arm_angle_deg": round(rv["arm_angle_deg"], 2), "split": str(split[c])}
    write_json(info, out / f"{slug}.clips.json")
    res = Path(a.results)
    res.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(res / "metrics.csv", index=False)

    pd.set_option("display.width", 220)
    print(f"\nheight {height_in} in; pre_s {pre_s}; release_index {ri}; foot_strike_index {fs}; "
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
    print(f"  z gap {dz:+.3f} ft -> "
          f"{'EXCEEDS 0.15 ft: FEET MAY FLOAT/SINK' if abs(dz) > 0.15 else 'within 0.15 ft'}")
    for nm, g in (("IN-SAMPLE", lab[lab["fresh"] == "no"]), ("FRESH", lab[lab["fresh"] == "yes"])):
        print(f"\nrelease detector vs labels {nm}: n={len(g)} median|err|={g['err'].abs().median()} "
              f"max|err|={g['err'].abs().max()} within2={(g['err'].abs() <= 2).sum()}/{len(g)}")


if __name__ == "__main__":
    main()
