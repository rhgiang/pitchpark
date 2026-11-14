"""
Anchor a recovered delivery to Statcast coordinates (ft; x toward catcher's right, y plate -> pitcher,
z up; rubber front edge y = 60.5, mound top z = 10/12) and nudge it to match mean release values.
"""
import numpy as np

RUBBER_Y_FT = 60.5
MOUND_Z_FT = 10 / 12
# SMPL joint indices (web/src/skeleton.js)
_PIVOT_FOOT = {"R": 11, "L": 10}  # toe joint of the throwing-side (pivot) foot
_ANKLE = {"L": 7, "R": 8}
_TOE = {"L": 10, "R": 11}
_SHOULDER = {"L": 16, "R": 17}
_ELBOW = {"L": 18, "R": 19}
_WRIST = {"L": 20, "R": 21}
_HAND = {"L": 22, "R": 23}  # knuckles: where the ball sits at release
_HEAD = 15

# Why 1.07: SMPL's "head" joint is near the base of the skull, ~7% of standing height below the crown
# (roughly 0.93 H; anthropometric tables, e.g. Winter, Biomechanics of Human Movement), and the toe joint
# is ~on the ground.  Multiplying head-to-foot by 1.07 approximates crown height.
_HEAD_FACTOR = 1.07


def _smoothstep(t: np.ndarray) -> np.ndarray:
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3 - 2 * t)


def _check(throws: str) -> None:
    if throws not in ("L", "R"):
        raise ValueError("throws must be 'L' or 'R'")


def _lock_feet(p_ft: np.ndarray, fps: float, v_lo_ftps: float = 0.5, v_hi_ftps: float = 1.5) -> np.ndarray:
    """Hold each foot's horizontal position while it is planted, blending in/out (no hard switch).

    Plant weight = 1 below v_lo, 0 above v_hi, smoothstep between (a swinging foot moves >> 3 ft/s, a
    planted one ~0; the band absorbs lift noise).  Within each planted run the foot is pulled to its mean
    position by weight w(t); only the ankle and toe move (a cm-scale shift, leg proportions unaffected).
    """
    out = p_ft.copy()
    for side in ("L", "R"):
        a = p_ft[:, _ANKLE[side], :2]
        v = np.linalg.norm(np.gradient(a, axis=0), axis=1) * fps
        v = np.convolve(v, np.ones(3) / 3, mode="same")
        w = 1.0 - _smoothstep((v - v_lo_ftps) / (v_hi_ftps - v_lo_ftps))
        planted = np.concatenate([[0], (w >= 0.5).astype(int), [0]])
        starts = np.nonzero(np.diff(planted) == 1)[0]
        ends = np.nonzero(np.diff(planted) == -1)[0]
        delta = np.zeros_like(a)
        for s, e in zip(starts, ends):  # run is frames [s, e)
            hold = a[s:e].mean(axis=0)
            lo, hi = s, e
            while lo > 0 and w[lo - 1] > 0:
                lo -= 1
            while hi < len(w) and w[hi] > 0:
                hi += 1
            delta[lo:hi] = w[lo:hi, None] * (hold - a[lo:hi])
        out[:, _ANKLE[side], :2] += delta
        out[:, _TOE[side], :2] += delta
    return out


def anchor(j_m: np.ndarray, up: np.ndarray, height_ft: float, throws: str, release_index: int,
           fps: float = 60.0) -> np.ndarray:
    """(F,24,3) metres in any frame with gravity-up `up` -> Statcast feet.

    Rotate so up = +z and the pelvis's horizontal forward at release = -y (toward the plate); scale so
    standing height (1.07 * median head-to-lower-foot over the first 0.2 s) = height_ft; lock planted feet
    (fps needed for foot speed; optional trailing arg since the brief's signature lacks it); translate so
    the pivot foot at frame 0 is at (0, 60.5, 10/12).
    """
    _check(throws)
    z = np.asarray(up, float) / np.linalg.norm(up)
    # SMPL: left = up x forward, so forward = left x up, with left = l_hip - r_hip.
    fwd = np.cross(j_m[release_index, 1] - j_m[release_index, 2], z)
    fwd -= z * (fwd @ z)
    fwd /= np.linalg.norm(fwd)
    y = -fwd
    x = np.cross(y, z)
    p = j_m @ np.stack([x, y, z], axis=1)

    n0 = max(2, int(round(0.2 * fps)))
    feet = p[:n0][:, [10, 11]]
    lower = feet[np.arange(n0), np.argmin(feet[..., 2], axis=1)]
    height_m = _HEAD_FACTOR * np.median(np.linalg.norm(p[:n0, _HEAD] - lower, axis=1))
    p = p * (height_ft / height_m)

    p = _lock_feet(p, fps)
    return p + (np.array([0.0, RUBBER_Y_FT, MOUND_Z_FT]) - p[0, _PIVOT_FOOT[throws]])


def release_values(j_ft: np.ndarray, release_index: int, throws: str) -> dict:
    """Release point (ft), extension (ft, 60.5 - hand y) and arm angle (deg) at `release_index`.

    Arm angle is the catcher's-view (frontal, x-z plane) angle of shoulder -> hand above horizontal,
    atan2(dz, |dx|), ignoring dy: 0 = sidearm, 90 = straight over the top.  This is Statcast's arm_angle
    convention and the one M4's web/src/pitcher.js uses (shoulder = hand - reach*(s*cos t, sin t, 0))."""
    _check(throws)
    hand = j_ft[release_index, _HAND[throws]]
    d = hand - j_ft[release_index, _SHOULDER[throws]]
    return {"release_x_ft": float(hand[0]), "release_z_ft": float(hand[2]),
            "extension_ft": float(RUBBER_Y_FT - hand[1]),
            "arm_angle_deg": float(np.degrees(np.arctan2(d[2], abs(d[0]))))}


def correct_to_mean(j_ft: np.ndarray, release_index: int, foot_strike_index: int, throws: str,
                    target: dict) -> np.ndarray:
    """One-time whole-body correction so release_values match `target` (to ~0.05 ft / 1 deg).

    Per pass: (1) rotate the throwing arm chain about the shoulder, in the catcher's-view (x-z) plane --
    the plane the arm angle is defined in (see release_values), i.e. about the y axis, y untouched, with the
    sense mirrored by the sign of the release dx so it works for L and R -- by the arm-angle gap, eased with smoothstep over foot_strike -> release and held after (no
    snap back in the follow-through); (2) translate everything by the x/z gap; (3) stretch y about the
    pivot foot at frame 0 so extension matches.  (1) moves the hand so passes repeat (converges in 1-2).  The angle step runs for gaps > 0.25 deg (not the brief's 2
    deg) so the final result is guaranteed within 1 deg.
    """
    _check(throws)
    j = np.array(j_ft, float)
    n = len(j)
    ease = _smoothstep((np.arange(n) - foot_strike_index) / max(release_index - foot_strike_index, 1))
    chain = [_ELBOW[throws], _WRIST[throws], _HAND[throws]]
    sh = _SHOULDER[throws]
    for _ in range(6):
        rv = release_values(j, release_index, throws)
        gap = target["arm_angle_deg"] - rv["arm_angle_deg"]
        if abs(gap) > 0.25:
            sgn = 1.0 if j[release_index, _HAND[throws], 0] - j[release_index, sh, 0] >= 0 else -1.0
            ang = np.radians(gap) * ease
            c, sn = np.cos(ang), np.sin(ang)
            for k in chain:  # in-plane rotation of (sgn*dx, dz) about the (moving) shoulder, per frame
                r = j[:, k] - j[:, sh]
                u, w = sgn * r[:, 0], r[:, 2]
                j[:, k, 0] = j[:, sh, 0] + sgn * (c * u - sn * w)
                j[:, k, 2] = j[:, sh, 2] + sn * u + c * w
        rv = release_values(j, release_index, throws)
        j[..., 0] += target["release_x_ft"] - rv["release_x_ft"]
        j[..., 2] += target["release_z_ft"] - rv["release_z_ft"]
        p0 = j[0, _PIVOT_FOOT[throws], 1]
        hand_y = j[release_index, _HAND[throws], 1]
        j[..., 1] = p0 + (j[..., 1] - p0) * ((RUBBER_Y_FT - target["extension_ft"] - p0) / (hand_y - p0))
        rv = release_values(j, release_index, throws)
        if (abs(rv["arm_angle_deg"] - target["arm_angle_deg"]) < 0.25
                and abs(rv["extension_ft"] - target["extension_ft"]) < 0.01):
            break
    return j
