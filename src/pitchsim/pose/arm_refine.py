"""2D-guided refinement of the throwing arm through the arm action (M5).

Problem (Sale, whose arm is nearly horizontal and sweeps across the image): the monocular lift cannot see
depth.  When the upper arm points at the centre-field camera, its lifted 3D bone collapses (1.04 ft -> 0.28 ft)
and its direction turns ~150 deg in two frames.  Averaging 5 clips does not fix it: all 5 lifts do the same.
But the broadcast image sees the catcher's-view plane (x, z) well, and that is the plane the horizontal
sweep lives in.  So the 2D keypoints of the TRAIN clips say where the arm is in the image; only the depth
(the component along the camera's viewing axis) is missing.

What this does, for the window foot strike - 0.2 s .. release + 0.2 s (full effect from fs - 0.1 s to release + 0.1 s):
  1. Per train clip, fit an affine camera (compare2d.fit_camera, the same one the 2D score uses) mapping the
     delivery's joints to that clip's keypoints, through the same time alignment the build uses (a PCHIP warp
     through foot strike and release).
  2. Rebuild the upper arm / forearm as unit directions u_f, v_f with the segment lengths held FIXED at their
     standing values (nothing shrinks).  A depth flip of a direction (reflection across the plane perpendicular
     to the viewing axis) leaves every projection unchanged: that is the sign ambiguity.
  3. Resolve it by continuity: a small Viterbi pass over the four flip combinations per frame, with a cost for
     turning between frames, a prior toward the current 3D direction (trusted only where the current bone
     is nearly full length, i.e. where depth was seen) and a penalty for the elbow's hinge flipping sign (a
     hyperextended elbow is a hinge reversal).
  4. Polish with a joint least squares: 2D data + smoothness (angular acceleration) + jerk of the joint paths
     + the prior + the hand staying on its corrected path + the hand pinned to the release point.
  5. Blend in and out at the window edges with smootherstep so nothing jumps.
Runs AFTER correct_to_mean, in the final frame: the correction rotates the arm in the x-z plane (the camera's
image plane) by up to ~20 deg, so a refinement done before it would be undone in exactly the plane the 2D
data constrains.  The hand at the release frame is pinned to the corrected value, so release values stay on target.
Where the tracks have no data (the blurred whip, a few frames after release) the path is the smooth
interpolation of its neighbours; the prior holds it near the lift where the lift was not collapsed.
Pure numpy / scipy.  Everything in the tests built here is SYNTHETIC.
"""
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from pitchsim.pose.anchor import _ELBOW, _HAND, _SHOULDER, _WRIST, _smootherstep
from pitchsim.pose.compare2d import (CONF_MIN, COCO_IDX, SMPL_IDX, fit_camera, sample_frames, shot_start,
                                     torso_px)
from pitchsim.pose.delivery import time_warp

# position of the throwing arm's shoulder / elbow / wrist in the 12-joint arrays of compare2d
ARM12 = {"L": (0, 2, 4), "R": (1, 3, 5)}


@dataclass
class Track:
    """One train clip's 2D keypoints, in the 12-joint order of compare2d (pixels), plus its landmarks."""
    xy: np.ndarray    # (F, 12, 2)
    w: np.ndarray     # (F, 12) bool: confident, and after the shot start
    rp: int           # release pose frame
    fs: int           # foot-strike frame
    fps: float


def track_from_pose(pose, rp: int, fs: int) -> Track:
    """Track from a pose2d.Pose2D (COCO-17).  Frames before the camera shot the release sits in are dropped,
    as in m5_compare2d (a batter close-up before the cut is not the pitcher)."""
    xy = pose.xy[:, COCO_IDX].astype(float)
    w = pose.conf[:, COCO_IDX] >= CONF_MIN
    w[:shot_start(xy, w, rp, pose.fps)] = False
    return Track(xy, w, rp, fs, float(pose.fps))


def release_reach_ratio(j_ft: np.ndarray, ri: int, throws: str) -> float:
    """Shoulder -> hand distance at the release frame over the full (standing) arm length: ~1 for a straight
    arm.  Well below 1 means the monocular lift foreshortened the arm at release (Sale 0.72, Misiorowski 0.85),
    which correct_to_mean then builds the body around (shoulder too close to the Statcast release point).
    Standing = median over the first 60 frames (the set position, arms in view), as in refine_arm."""
    sh, el, wr, ha = _SHOULDER[throws], _ELBOW[throws], _WRIST[throws], _HAND[throws]
    seg = lambda a, b: float(np.median(np.linalg.norm(j_ft[:60, b] - j_ft[:60, a], axis=1)))  # noqa: E731
    return float(np.linalg.norm(j_ft[ri, ha] - j_ft[ri, sh]) / (seg(sh, el) + seg(el, wr) + seg(wr, ha)))


def _unit(a):
    return a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-9)


def _rot_between(a, b, x):
    """Rotate vectors x by the rotation taking unit a to unit b (Rodrigues), row-wise (N, 3)."""
    v = np.cross(a, b)
    c = np.sum(a * b, -1, keepdims=True)
    # (1 + c) -> 0 only when b = -a (a 180 deg flip of the forearm); then x just flips sign, which is as
    # good a hand as any and avoids a division by zero.
    k = 1.0 / np.maximum(1.0 + c, 1e-6)
    vx = np.cross(v, x)
    return x + vx + k * np.cross(v, vx)


def _smooth_penalty(d, lam):
    """Second difference of a (F, 3) direction sequence: the angular acceleration per frame^2."""
    return lam * (d[2:] - 2 * d[1:-1] + d[:-2]).ravel()


def refine_arm(j_ft: np.ndarray, ri: int, fsi: int, throws: str, tracks: list[Track], fps: float = 60.0,
               smooth: float = 0.3, prior: float = 0.5, pin: float = 100.0, hand_w: float = 1.0, jerk_w: float = 0.5, jump: float = 0.05,
               data_w: float = 1.0, info: dict | None = None) -> np.ndarray:
    """Return j_ft with the throwing arm's elbow, wrist and hand re-solved against the tracks' 2D keypoints.

    j_ft (T, 24, 3) in Statcast ft (the FINAL delivery); ri / fsi release and foot-strike frame indices;
    tracks: the train clips' 2D (the hold-out rule: never pass test clips).
    smooth: weight of angular acceleration of each bone direction (residual = smooth * d2u, in unit-vector
      units per frame^2); prior: weight toward the current direction, scaled by how far from collapsed the
      current bone is (0 at length 0, 1 at full length); hand_w: weight (per ft) of the hand staying on its
      corrected path; jerk_w: weight on the third difference of the elbow / wrist / hand paths (ft per frame^3:
      0.08 is the smoothed delivery's own level, noisy 2D tracks give 0.3+); pin: weight (per ft) of the hand at the release frame;
      jump: Viterbi cost per rad^2 turned between frames; data_w: weight of the 2D residuals (torso lengths) against
      the priors (torso lengths against unit vectors; raising it makes the 2D win over the lift prior).
    Segment lengths are the standing values (median over the set frames, hence 'fixed'); info (optional out-dict) gets diagnostics.
    """
    j = np.array(j_ft, float)
    T = len(j)
    sh, el, wr, ha = _SHOULDER[throws], _ELBOW[throws], _WRIST[throws], _HAND[throws]
    a_s, a_e, a_w = ARM12[throws]
    # Segment lengths: the STANDING values, the median over the set / windup frames (foot strike - 0.6 s and
    # earlier), when the arms are in front of the camera and the lift sees them at full length.  Not the
    # release frame's: at release the forearm points along the camera's axis and is depth-compressed
    # (Sale: 0.70 ft at release against 1.04 ft standing).  Held fixed through the window, so no bone shrinks.
    base = slice(0, max(fsi - int(0.6 * fps), 5))
    seg = lambda a, b: float(np.median(np.linalg.norm(j[base, b] - j[base, a], axis=1)))  # noqa: E731
    Lu, Lf, Lh = seg(sh, el), seg(el, wr), seg(wr, ha)
    pad, ramp = int(round(0.2 * fps)), int(round(0.1 * fps))
    lo, hi = max(fsi - pad, 0), min(ri + pad, T - 1)
    fr = np.arange(lo, hi + 1)
    n = len(fr)
    # blend weight: 1 on [fs - 0.1 s, release + 0.1 s], smootherstep to 0 at the window edges
    wb = _smootherstep(np.minimum((fr - lo) / ramp, (hi - fr) / ramp))
    S = j[fr, sh]
    u0, v0 = _unit(j[fr, el] - S), _unit(j[fr, wr] - j[fr, el])
    # how well the current bone was seen: its length / the full length (a depth-collapsed bone is short)
    cu = np.clip(np.linalg.norm(j[fr, el] - S, axis=1) / Lu, 0, 1)
    cf = np.clip(np.linalg.norm(j[fr, wr] - j[fr, el], axis=1) / Lf, 0, 1)

    # ---- per-track camera + targets in the delivery's time ----------------------------------------------
    rep = j[:, SMPL_IDX]
    s0, s1 = max(fsi - int(0.5 * fps), 0), min(ri + int(0.5 * fps), T - 1)   # span for the camera fit
    span = np.arange(s0, s1 + 1)
    data = []   # per track: A (2,3), b (n,2), xe (n,2), xw (n,2), we (n,), ww (n,), torso
    for tr in tracks:
        tau = (span - ri) / fps
        # PCHIP warp through foot strike and release, the same landmarks the build's average uses
        warp = time_warp(np.array([(fsi - ri) / fps, 0.0]), np.array([(tr.fs - tr.rp) / tr.fps, 0.0]))
        pos = tr.rp + warp(tau) * tr.fps
        x = sample_frames(tr.xy, pos)
        w = sample_frames(tr.w.astype(float), pos) > 0.99     # a frame is used only if both neighbours were
        w &= np.isfinite(x).all(-1)
        A, b, ok = fit_camera(rep[span], x, w)
        if ok.sum() < 6:
            continue
        sel = np.searchsorted(span, fr)
        torso = torso_px(x, w)
        wk = w[sel] & ok[sel][:, None]
        data.append((A, np.nan_to_num(b[sel]), x[sel, a_e], x[sel, a_w], wk[:, a_e], wk[:, a_w], torso))
    if not data:
        return j   # nothing to guide the arm: leave it alone
    # viewing axis: the null direction of each camera's 2x3 matrix (the depth that cannot be seen)
    nv = [np.cross(A[0], A[1]) for A, *_ in data]
    nv = [v / np.linalg.norm(v) * (1 if v @ nv[0] >= 0 else -1) for v in nv]
    nax = _unit(np.mean(nv, axis=0))

    def data_res(E, W):
        """2D residuals in torso lengths, all tracks, elbow and wrist: (F, n_tracks, 2 joints, 2)."""
        out = np.zeros((len(E), len(data), 2, 2))
        for k, (A, b, xe, xw, we, ww, tor) in enumerate(data):
            out[:, k, 0] = np.where(we[:, None], (E @ A.T + b - xe) / tor, 0.0)
            out[:, k, 1] = np.where(ww[:, None], (W @ A.T + b - xw) / tor, 0.0)
        return out

    hd0 = _unit(j[fr, ha] - j[fr, wr])

    def arm(u, v):
        E = S + Lu * u
        return E, E + Lf * v

    def hand(W, v, a=0, b=None):
        """Hand: fixed length Lh, rigid on the forearm (turned by the same rotation as the forearm)."""
        return W + Lh * _rot_between(v0[a:b], v, hd0[a:b])

    rel = int(ri - lo)
    # ---- candidates: 4 depth-flip combinations per frame, each polished on this frame's data alone --------
    flip = lambda d: d - 2 * (d @ nax)[:, None] * nax  # noqa: E731  reflection across the plane perpendicular to the view
    cost = np.zeros((n, 4))
    cand = np.zeros((n, 4, 2, 3))
    def frame_res(i, u, v):
        """2D residuals (torso lengths) of frame i for directions u, v: (n_tracks, 2 joints, 2)."""
        E = S[i] + Lu * u
        W = E + Lf * v
        d = np.zeros((len(data), 2, 2))
        for k, (A, b, xe, xw, we, ww, tor) in enumerate(data):
            if we[i]:
                d[k, 0] = (A @ E + b[i] - xe[i]) / tor
            if ww[i]:
                d[k, 1] = (A @ W + b[i] - xw[i]) / tor
        return d

    for c, (fu, fv) in enumerate(((0, 0), (1, 0), (0, 1), (1, 1))):
        ua = flip(u0) if fu else u0
        va = flip(v0) if fv else v0
        for i in range(n):
            def res(p, i=i):
                u, v = _unit(p[:3]), _unit(p[3:])
                r_i = [frame_res(i, u, v).ravel(), 0.02 * (u - u0[i]), 0.02 * (v - v0[i])]
                if i == rel:   # the release frame carries the hand pin from the start
                    Wi = S[i] + Lu * u + Lf * v
                    r_i.append(pin * (hand(Wi[None], v[None], i, i + 1)[0] - j[ri, ha]))
                return np.concatenate(r_i)
            r = least_squares(res, np.concatenate([ua[i], va[i]]))
            cand[i, c, 0], cand[i, c, 1] = _unit(r.x[:3]), _unit(r.x[3:])
            u, v = cand[i, c]
            # state cost: this frame's 2D misfit + prior toward the current direction where it was seen
            cost[i, c] = (frame_res(i, u, v) ** 2).sum() + prior ** 2 * (
                cu[i] ** 2 * np.sum((u - u0[i]) ** 2) + cf[i] ** 2 * np.sum((v - v0[i]) ** 2))

    def hinge(u, v):
        h = np.cross(u, v)
        s = np.linalg.norm(h)
        return h / max(s, 1e-9), s

    ang = lambda a, b: np.arccos(np.clip(np.sum(a * b, -1), -1, 1))  # noqa: E731
    best = cost.copy()
    back = np.zeros((n, 4), int)
    for i in range(1, n):
        for c in range(4):
            tot = np.empty(4)
            for p in range(4):
                up, vp = cand[i - 1, p]
                uc, vc = cand[i, c]
                hp, sp = hinge(up, vp)
                hc, sc = hinge(uc, vc)
                # a hinge reversal with a clearly bent elbow = the elbow bending the wrong way (hyperextension)
                rev = 0.5 * max(-hp @ hc, 0.0) * min(sp, sc)
                tot[p] = best[i - 1, p] + jump * (ang(up, uc) ** 2 + ang(vp, vc) ** 2) + rev
            back[i, c] = int(np.argmin(tot))
            best[i, c] += tot[back[i, c]]
    pick = np.zeros(n, int)
    pick[-1] = int(np.argmin(best[-1]))
    for i in range(n - 1, 0, -1):
        pick[i - 1] = back[i, pick[i]]
    u_init = np.array([cand[i, pick[i], 0] for i in range(n)])
    v_init = np.array([cand[i, pick[i], 1] for i in range(n)])

    # ---- joint polish: data + smoothness + prior + release pin -------------------------------------------
    def residuals(p):
        u, v = _unit(p[:n * 3].reshape(n, 3)), _unit(p[n * 3:].reshape(n, 3))
        E, W = arm(u, v)
        return np.concatenate([
            data_w * data_res(E, W).ravel(),
            _smooth_penalty(u, smooth), _smooth_penalty(v, smooth),
            # 3rd difference of the joint paths: the build smooths the average (Savitzky-Golay 0.15 s) and this must not
            # undo that with the frame-to-frame noise of the 2D tracks (~4x the jerk without it)
            jerk_w * np.diff(E, 3, axis=0).ravel(), jerk_w * np.diff(W, 3, axis=0).ravel(),
            jerk_w * np.diff(hand(W, v), 3, axis=0).ravel(),
            (prior * cu[:, None] * (u - u0)).ravel(), (prior * cf[:, None] * (v - v0)).ravel(),
            # hand path: where the corrected delivery has the hand (the hand's own 3D path is as good as the lift
            # is; the 2D data cannot say where it is in depth), trusted only where both bones were seen (4th power of the shorter bone's length ratio: a half-collapsed arm is
            # already a wrong hand).  Without
            # it the hand drifts in the frames before the pinned release and then snaps onto it.
            (hand_w * (np.minimum(cu, cf) ** 4)[:, None] * (hand(W, v) - j[fr, ha])).ravel(),
            # release pin: the HAND stays where correct_to_mean put it (the bones are longer now, so the
            # directions may differ there, but the release point is the target)
            pin * (hand(W, v)[rel] - j[ri, ha])])

    sol = least_squares(residuals, np.concatenate([u_init.ravel(), v_init.ravel()]))
    u, v = _unit(sol.x[:n * 3].reshape(n, 3)), _unit(sol.x[n * 3:].reshape(n, 3))
    E, W = arm(u, v)
    H = hand(W, v)
    out = j.copy()
    for k, new in ((el, E), (wr, W), (ha, H)):
        out[fr, k] = j[fr, k] + wb[:, None] * (new - j[fr, k])
    if info is not None:
        d_before = data_res(j[fr, el], j[fr, wr])
        d_after = data_res(out[fr, el], out[fr, wr])
        info.update(lengths_ft=[Lu, Lf, Lh], window=[int(lo), int(hi)], n_tracks=len(data),
                    data_rms_before=float(np.sqrt(np.mean(d_before ** 2))),
                    data_rms_after=float(np.sqrt(np.mean(d_after ** 2))),
                    rms_by_frame_before=np.sqrt((d_before ** 2).mean((1, 2, 3))).round(3).tolist(),
                    rms_by_frame_after=np.sqrt((d_after ** 2).mean((1, 2, 3))).round(3).tolist(),
                    tracks_with_elbow_by_frame=np.sum([t[4] for t in data], 0).tolist(),
                    tracks_with_wrist_by_frame=np.sum([t[5] for t in data], 0).tolist(),
                    flipped_frames=int(np.sum(pick != 0)), viewing_axis=nax.tolist(),
                    upper_arm_len_min_before_ft=float(np.linalg.norm(j[fr, el] - S, axis=1).min()),
                    upper_arm_len_min_after_ft=float(np.linalg.norm(out[fr, el] - S, axis=1).min()))
    return out
