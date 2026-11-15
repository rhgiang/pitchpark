"""
One delivery from many clips: landmarks, time-warp alignment, averaging, smoothing.

Frame-agnostic: nothing here assumes which axis is up except `landmarks`, which takes `up` explicitly.
"""
import numpy as np
from scipy.interpolate import PchipInterpolator
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


def average_delivery(seqs: list[np.ndarray], marks: list[dict], pre_s: float, fps_out: float = 60.0,
                     post_s: float = 0.6, excluded: list | None = None) -> tuple[np.ndarray, int]:
    """Time-warp clips so their landmarks coincide, average, smooth.  Returns (joints (T,24,3), release_index).

    marks[i]: dict with leg_lift, foot_strike, release (frame indices, from `landmarks`) and "fps" (that
      clip's frame rate; the brief's signature carries no per-clip fps, so it rides in the marks).  Any
      other int entries (e.g. "rocker") are extra landmark knots; every clip must have the same keys.
    pre_s / post_s: seconds kept before / after release.  release_index == round(pre_s * fps_out).
    excluded: optional out-parameter; a clip that does not cover [release - pre_s, release + post_s] is
      appended as (clip_index, "short").  An out-list keeps the return a plain 2-tuple as the brief says.
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
        acc += s[lo] * (1 - f) + s[hi] * f
    if used == 0:
        raise ValueError("no clip covers the requested window after time-warping")
    avg = acc / used

    # Savitzky-Golay (Savitzky & Golay 1964): local cubic fit over ~0.15 s.  Cubic keeps the sharp
    # release-speed peak (a moving average would flatten it); 0.15 s is shorter than the arm-acceleration
    # phase but long enough to kill frame-to-frame lift noise.
    # window must exceed the polynomial order (4) and fit in the signal
    if n_out < 5:
        raise ValueError(f"output has only {n_out} frames; need >= 5 for Savitzky-Golay smoothing")
    win = max(int(round(0.15 * fps_out)) | 1, 5)
    win = min(win, n_out if n_out % 2 else n_out - 1)
    return savgol_filter(avg, win, 3, axis=0), ri


def ground_root(j: np.ndarray, throws: str, foot_strike: int | None = None) -> np.ndarray:
    """Replace the lift's root translation with foot-planting, in the same frame as `j`.

    Why: the monocular lift gets the body's pose relative to itself right but its position in the camera
    frame wrong, especially depth (the real clips' pelvis depth jumps by 5-10 m between frames, which would
    put the pitcher tens of feet from the rubber).  A pitcher's feet are planted, so use them as the world
    reference: the pivot (throwing-side) foot stays where it started until foot strike, then the lead foot
    stays where it landed.  Both are planted at foot strike, so the hand-over is continuous.
    foot_strike=None grounds on the pivot foot for every frame (used to find foot strike itself).
    The toe is used raw (not filtered): filtering it would leave the root noise in the body; any toe jitter
    this copies into the body is removed by the averaging and Savitzky-Golay smoothing downstream.
    """
    if throws not in ("L", "R"):
        raise ValueError("throws must be 'L' or 'R'")
    j = np.asarray(j, float)
    piv_i, lead_i = _TOE[throws], _TOE["L" if throws == "R" else "R"]
    out = j - j[:, piv_i][:, None, :] + j[0, piv_i]
    if foot_strike is not None:
        lead = out[:, lead_i].copy()
        out[foot_strike:] = out[foot_strike:] - lead[foot_strike:, None, :] + lead[foot_strike]
    return out
