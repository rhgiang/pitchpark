"""
One delivery from many clips: landmarks, time-warp alignment, averaging, smoothing.

Frame-agnostic: nothing here assumes which axis is up except `landmarks`, which takes `up` explicitly.
"""
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter

# SMPL joint indices (same order as JOINTS in web/src/skeleton.js)
_KNEE = {"L": 4, "R": 5}
_ANKLE = {"L": 7, "R": 8}
_TOE = {"L": 10, "R": 11}


def _lead(throws: str) -> str:
    """Lead (glove-side) leg is the side opposite the throwing arm."""
    if throws not in ("L", "R"):
        raise ValueError("throws must be 'L' or 'R'")
    return "L" if throws == "R" else "R"


def landmarks(j: np.ndarray, release: int, throws: str, up: np.ndarray | None = None) -> dict[str, int]:
    """Frame indices of leg_lift, foot_strike and release for one clip.

    j: (F, 24, 3) joints, any consistent unit.  up: gravity-up unit vector in j's frame; defaults to +z
    (Statcast frame), pass `Lift3D.up` for raw camera-frame lifts (the brief's signature has no `up`; the
    knee "height" needs it, so it is an optional trailing argument).
    leg_lift: frame of peak lead-knee height before release.
    foot_strike: first frame after the lead ankle's peak speed where its speed falls below 10% of that peak.
      Starting from the peak matters: at the top of the leg lift the ankle is also nearly still, so "first
      frame after leg_lift below 10% of peak" would fire immediately.
    """
    up = np.array([0.0, 0.0, 1.0]) if up is None else np.asarray(up, float) / np.linalg.norm(up)
    lead = _lead(throws)
    knee_h = j[: release + 1, _KNEE[lead]] @ up
    leg_lift = int(np.argmax(knee_h))
    ank = j[:, _ANKLE[lead]]
    speed = np.linalg.norm(np.gradient(ank, axis=0), axis=1)
    speed = np.convolve(speed, np.ones(3) / 3, mode="same")  # light smoothing: per-frame lift noise
    seg = speed[leg_lift: release + 1]
    pk = int(np.argmax(seg))
    below = np.nonzero(seg[pk:] < 0.1 * seg[pk])[0]
    fs = leg_lift + pk + (int(below[0]) if below.size else int(np.argmin(seg[pk:])))
    fs = int(np.clip(fs, leg_lift + 1, release - 1))
    return {"leg_lift": leg_lift, "foot_strike": fs, "release": int(release)}


_LEGS = [1, 4, 7, 10, 2, 5, 8, 11]          # L hip, knee, ankle, toe, then R
_LEGS_SWAPPED = [2, 5, 8, 11, 1, 4, 7, 10]


def fix_leg_swaps(j: np.ndarray, anchor_frame: int, margin: float = 0.1) -> tuple[np.ndarray, int]:
    """Undo left/right leg swaps in a lift.  Returns (repaired copy, number of frames swapped back).

    Why: a monocular lift sometimes labels the left leg as the right for a stretch of frames, then snaps
    back -- a one-frame jump of every leg joint (0.5-0.9 m in 5 of the 14 Ohtani train clips, all before
    the leg lift).  Averaged in, the snap is a hitch in the delivery at the leg lift, and the swapped
    stretch hides the rocker step (its "lead toe" is really the pivot toe).
    Walk out from `anchor_frame` (a frame trusted to be right, e.g. release), each frame compared with its
    already-repaired neighbour: if the swapped labelling is closer by more than `margin` (j's units; the
    farthest leg joint, pelvis-relative so root noise does not count), swap it.  Real leg motion moves a
    joint < 0.1 m per frame at 60 fps, while a swap moves the hips' width plus the stride.
    """
    j = np.array(j, float)
    rel = j - j[:, :1]
    n_sw = 0
    for frames, step in ((range(anchor_frame - 1, -1, -1), 1), (range(anchor_frame + 1, len(j)), -1)):
        for f in frames:
            g = rel[f + step, _LEGS]
            keep = np.linalg.norm(rel[f, _LEGS] - g, axis=1).max()
            swap = np.linalg.norm(rel[f, _LEGS_SWAPPED] - g, axis=1).max()
            if swap + margin < keep:
                j[f, _LEGS] = j[f, _LEGS_SWAPPED]
                rel[f, _LEGS] = rel[f, _LEGS_SWAPPED]
                n_sw += 1
    return j, n_sw


def lift_snap(j: np.ndarray, release: int, fps: float, jump: float = 0.4) -> int | None:
    """First frame after the last whole-leg snap before release - 0.3 s, or None.

    A snap is a frame where some leg joint (pelvis-relative, so root noise does not count) jumps more than
    `jump` (j's units; 0.4 m) from the frame before.  Real leg joints move < 0.1 m per frame at 60 fps
    away from release; in 3 Ohtani train clips the lift sits frozen in the set stance and then snaps
    0.5-0.9 m to the leg lift in one frame.  Frames before the snap are not the pitcher's motion.  The last
    0.3 s before release is not searched: the stride and arm are fast there and leg jumps of 0.2-0.34 m
    are real.
    """
    rel = j[:, _LEGS] - j[:, :1]
    hi = release - int(round(0.3 * fps))
    d = np.linalg.norm(np.diff(rel[: hi + 1], axis=0), axis=2).max(1)
    big = np.nonzero(d > jump)[0]
    return int(big[-1] + 1) if big.size else None


def rocker_step(j: np.ndarray, throws: str, leg_lift: int, fps: float, up: np.ndarray | None = None,
                min_step: float = 0.6) -> int | None:
    """Frame of the rocker step (the lead foot's small step sideways / back before the leg lift), or None.

    Ohtani starts from a set stance, steps the lead foot ~1.3 ft sideways and sways his hips over it,
    then lifts the leg.  The step's plateau is the lead toe's horizontal offset from frame 0 over
    leg_lift - 0.8 s .. leg_lift - 0.6 s (planted after the step, before the leg starts to rise); the
    rocker step is the first frame the toe has covered half of it along that direction.  None when the
    plateau is under `min_step` (j's units; 0.6 ft by default: no visible step, or a lift that missed it).
    """
    up = np.array([0.0, 0.0, 1.0]) if up is None else np.asarray(up, float) / np.linalg.norm(up)
    d = j[:, _TOE[_lead(throws)]] - j[0, _TOE[_lead(throws)]]
    d = d - np.outer(d @ up, up)
    lo, hi = max(leg_lift - int(round(0.8 * fps)), 0), max(leg_lift - int(round(0.6 * fps)), 1)
    plateau = np.median(d[lo:hi], axis=0)
    size = np.linalg.norm(plateau)
    if size < min_step:
        return None
    along = d[:hi] @ (plateau / size)
    return int(np.argmax(along >= 0.5 * size))


def jitter(j: np.ndarray, fps: float) -> float:
    """RMS over joints and frames of the third time-difference (jerk), in units/s^3.
    Noise is amplified hard by differencing, so this is a sensitive jitter meter."""
    d3 = np.diff(np.asarray(j, float), n=3, axis=0) * fps ** 3
    return float(np.sqrt(np.mean(d3 ** 2)))


def time_warp(tmpl_s: np.ndarray, clip_s: np.ndarray) -> PchipInterpolator:
    """Monotone map from template time to a clip's time (s from release) through the landmark knots.

    tmpl_s / clip_s: increasing landmark times (template, this clip), release last.  Knots at -10 s / +10 s
    with slope-1 offsets keep the wind-up and follow-through at their natural speed.
    Why PCHIP (Fritsch & Carlson 1980) and not piecewise-linear: a linear warp changes slope at each
    landmark, so the warped clip's speed jumps there (a visible hitch at leg lift / foot strike).  PCHIP is
    a cubic through the same knots with continuous slope, and it never overshoots, so time stays monotone.
    """
    kx = np.concatenate([[tmpl_s[0] - 10], tmpl_s, [10.0]])
    ky = np.concatenate([[clip_s[0] - 10], clip_s, [10.0]])
    return PchipInterpolator(kx, ky)


def phase_smooth(j: np.ndarray, fps: float, slow_until: int, sigma_s: float = 0.05,
                 blend_s: float = 0.15) -> np.ndarray:
    """Extra smoothing for the slow part of a delivery (frames before `slow_until`), none after.

    Why: before the leg lift the body moves slowly (set stance, rocker step, hands coming together), so
    the averaged lifts' residual frame noise is a large share of the motion there and reads as a wobble
    (torso ~117 deg/s of angular noise at t ~ -1.5 s); the same smoothing through the stride and arm
    action would blunt real fast motion.  The output is a Gaussian-smoothed copy (sigma `sigma_s`, 0.05 s
    = 3 frames: the rocker step takes ~0.4 s, so it survives) blended into the input with a weight that
    is 1 up to `slow_until` - `blend_s` and eases (smootherstep) to 0 at `slow_until`.
    """
    j = np.asarray(j, float)
    smooth = gaussian_filter1d(j, sigma_s * fps, axis=0, mode="nearest")
    u = np.clip((slow_until - np.arange(len(j))) / max(blend_s * fps, 1.0), 0.0, 1.0)
    w = u ** 3 * (u * (6 * u - 15) + 10)
    return j + w[:, None, None] * (smooth - j)


def average_delivery(seqs: list[np.ndarray], marks: list[dict], pre_s: float, fps_out: float = 60.0,
                     post_s: float = 0.6, excluded: list | None = None,
                     weights: list | None = None, valid_from: list | None = None,
                     fade_s: float = 0.3) -> tuple[np.ndarray, int]:
    """Time-warp clips so their landmarks coincide, average, smooth.  Returns (joints (T,24,3), release_index).

    marks[i]: dict with leg_lift, foot_strike, release (frame indices, from `landmarks`) and "fps" (that
      clip's frame rate; the brief's signature carries no per-clip fps, so it rides in the marks).  Any
      other int entries (e.g. "rocker") are extra landmark knots; every clip must have the same keys.
    pre_s / post_s: seconds kept before / after release.  release_index == round(pre_s * fps_out).
    excluded: optional out-parameter; a clip that does not cover [release - pre_s, release + post_s] is
      appended as (clip_index, "short").  An out-list keeps the return a plain 2-tuple as the brief says.
    weights: optional per-clip f(tau) -> weight in [0, 1] (tau: output time, s from release; None = 1), so a
      clip can sit out part of the delivery (e.g. a clip whose lift missed the rocker step fades in only
      before the leg lift); the average divides by the summed weight.
    valid_from: optional per-clip first valid frame (e.g. `lift_snap`; None = all valid).  The clip's weight
      is 0 before it and eases in (smootherstep) over the next `fade_s` of clip time: a clip that joins
      the average abruptly changes the mix in one frame, which is itself a jump.
    Warp: `time_warp` (smooth, monotone) through the landmarks -> their median times relative to release.
    """
    keep = []
    for i, (s, m) in enumerate(zip(seqs, marks)):
        if "fps" not in m:
            raise ValueError(f"marks[{i}] needs an 'fps' entry")
        fps = m["fps"]
        if m["release"] / fps < pre_s or (len(s) - 1 - m["release"]) / fps < post_s:
            if excluded is not None:
                excluded.append((i, "short"))
        else:
            keep.append(i)
    if not keep:
        raise ValueError("no clip covers the requested window")

    # landmark times relative to release, seconds; template = median over clips
    keys = [k for k in marks[keep[0]] if k not in ("fps", "release")]
    rel_t = np.array([[(marks[i][k] - marks[i]["release"]) / marks[i]["fps"] for k in keys] + [0.0]
                      for i in keep])
    tmpl = np.median(rel_t, axis=0)
    order = np.argsort(tmpl, kind="stable")
    tmpl, rel_t = tmpl[order], rel_t[:, order]

    ri = int(round(pre_s * fps_out))
    n_out = ri + int(round(post_s * fps_out)) + 1
    tau = (np.arange(n_out) - ri) / fps_out
    acc = np.zeros((n_out, 24, 3))
    wsum = np.zeros(n_out)
    used = 0
    for c, i in enumerate(keep):
        s = np.asarray(seqs[i], float)
        if np.any(np.diff(rel_t[c]) <= 0):  # landmarks out of order: no monotone warp exists
            if excluded is not None:
                excluded.append((i, "landmark_order"))
            continue
        pos = marks[i]["release"] + time_warp(tmpl, rel_t[c])(tau) * marks[i]["fps"]  # fractional clip frames
        # Coverage on the WARPED positions: a clip whose wind-up is longer than the template's needs frames
        # before frame 0 once stretched; clamping would contribute frozen frames, so exclude it instead.
        if pos.min() < 0 or pos.max() > len(s) - 1:
            if excluded is not None:
                excluded.append((i, "short"))
            continue
        used += 1
        lo = np.floor(pos).astype(int)
        hi = np.minimum(lo + 1, len(s) - 1)
        f = (pos - lo)[:, None, None]
        wt = np.ones(n_out) if weights is None or weights[i] is None else np.asarray(weights[i](tau), float)
        if valid_from is not None and valid_from[i] is not None:
            u = np.clip((pos - valid_from[i]) / (fade_s * marks[i]["fps"]), 0.0, 1.0)
            wt = wt * u ** 3 * (u * (6 * u - 15) + 10)   # smootherstep: no jolt where the clip joins
        acc += wt[:, None, None] * (s[lo] * (1 - f) + s[hi] * f)
        wsum += wt
    if used == 0:
        raise ValueError("no clip covers the requested window after time-warping")
    if np.any(wsum <= 0):
        raise ValueError("clip weights sum to zero somewhere in the window")
    avg = acc / wsum[:, None, None]

    # Savitzky-Golay (Savitzky & Golay 1964): local cubic fit over ~0.15 s.  Cubic keeps the sharp
    # release-speed peak (a moving average would flatten it); 0.15 s is shorter than the arm-acceleration
    # phase but long enough to kill frame-to-frame lift noise.
    # window must exceed the polynomial order (4) and fit in the signal
    if n_out < 5:
        raise ValueError(f"output has only {n_out} frames; need >= 5 for Savitzky-Golay smoothing")
    win = max(int(round(0.15 * fps_out)) | 1, 5)
    win = min(win, n_out if n_out % 2 else n_out - 1)
    return savgol_filter(avg, win, 3, axis=0), ri


def ground_root(j: np.ndarray, throws: str, foot_strike: int | None = None,
                smooth_frames: int = 9, bridge_pre: int = 6, bridge_post: int = 6) -> np.ndarray:
    """Replace the lift's root translation with foot-planting, in the same frame as `j`.

    Why: the monocular lift gets the body's pose relative to itself right but its position in the camera
    frame wrong, especially depth (the real clips' pelvis depth jumps by 5-10 m between frames, which would
    put the pitcher tens of feet from the rubber).  A pitcher's feet are planted, so use them as the world
    reference: the pivot (throwing-side) foot stays where it started until foot strike, then the lead foot
    stays where it landed.  Both are planted at foot strike, so the hand-over is continuous.
    foot_strike=None grounds on the pivot foot for every frame (used to find foot strike itself).
    The toe's offset from the pelvis is smoothed over `smooth_frames` (see below); the planted toe joint
    keeps its own small jitter, which lock_feet removes.
    """
    if throws not in ("L", "R"):
        raise ValueError("throws must be 'L' or 'R'")
    j = np.asarray(j, float)
    piv_i, lead_i = _TOE[throws], _TOE["L" if throws == "R" else "R"]
    # The body is placed by its pelvis: world pelvis = planted toe - (toe - pelvis).  (toe - pelvis) is body-
    # relative, so the lift's root noise cancels in it and it can be smoothed (Savitzky-Golay, cubic, 9
    # frames ~ 0.15 s) without letting root noise back in.  Subtracting the raw toe instead handed the
    # toe joint's own frame-to-frame jitter to every joint: a whole-body kink 0.27 s before release in the
    # average (13 clips' pivot toes wobble as the heel comes up).
    def rel(k):
        r = j[:, k] - j[:, 0]
        w = min(smooth_frames, len(j) if len(j) % 2 else len(j) - 1)
        return savgol_filter(r, w, 3, axis=0) if w > 3 else r
    body = j - j[:, :1]
    r_piv = rel(piv_i)
    out = body - r_piv[:, None, :]
    out += j[0, piv_i] - out[0, piv_i]
    if foot_strike is not None:
        r_lead = rel(lead_i)
        fs = foot_strike
        land = out[fs, 0] + r_lead[fs]                       # where the lead toe is at foot strike
        out[fs:] = body[fs:] - r_lead[fs:, None, :] + land
        # Bridge the hand-over: before foot strike the lift's pivot toe slides toward the plate relative to
        # the body as the back heel comes up, so pinning it drags the whole body backward, and the switch to
        # the lead foot then snaps it forward (a kink in every joint ~0.27 s before release in the average).
        # Neither foot is a trustworthy anchor right there, so the root path from foot strike - bridge_pre
        # to foot strike + bridge_post is replaced by a cubic Hermite matching position and velocity at both
        # ends (C1), and every joint moves with it.  6 + 6 frames (0.1 s each side at 60 fps): a 15-frame
        # bridge also removed the kink but moved the pelvis so far ahead of the planted back foot at foot
        # strike that correct_to_mean had to drop the pelvis 1.4 ft and the back knee snapped 34 deg/frame.
        a, b = max(fs - bridge_pre, 1), min(fs + bridge_post, len(j) - 2)
        if b > a:
            P = out[:, 0]
            va, vb = P[a] - P[a - 1], P[b + 1] - P[b]          # per frame
            h = b - a
            u = (np.arange(a, b + 1) - a)[:, None] / h
            H = ((2 * u ** 3 - 3 * u ** 2 + 1) * P[a] + (u ** 3 - 2 * u ** 2 + u) * h * va
                 + (-2 * u ** 3 + 3 * u ** 2) * P[b] + (u ** 3 - u ** 2) * h * vb)
            out[a:b + 1] += (H - P[a:b + 1])[:, None, :]
    return out
