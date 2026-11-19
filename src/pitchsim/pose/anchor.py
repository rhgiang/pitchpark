"""
Anchor a recovered delivery to Statcast coordinates (ft; x toward catcher's right, y plate -> pitcher,
z up; rubber front edge y = 60.5, mound top z = 10/12) and nudge it to match mean release values.
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d, maximum_filter1d

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
_HIP = {"L": 1, "R": 2}
_KNEE = {"L": 4, "R": 5}
# Parent of each SMPL joint (JOINTS[].parent in web/src/skeleton.js).
SMPL_PARENT = (-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21)
# Joints that move with the pelvis: pelvis, hips, spine, neck, head, both arms.
_UPPER = [0, 1, 2, 3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
# The part of those that bends forward with the trunk (everything above the pelvis except the hips).
_TRUNK = [3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
# Cap on ADDED trunk flexion.  The averaged lift already leans ~30 deg forward at release (pelvis -> neck);
# pitchers' forward trunk tilt at ball release is roughly 30-50 deg (biomechanics reports, e.g. ASMI /
# Driveline), so +20 deg keeps the lean plausible.  The cap also keeps the solve well-posed: an upright
# trunk lowers the shoulder only ~(1 - cos) per degree, and flexing drags the arm angle down for the arm
# rotation to undo, so an uncapped flexion ran to 45-65 deg on synthetic cases.
MAX_ADDED_FLEXION_DEG = 20.0
# Mound cone, as in web/src/field.js (OBR Appendix 2: 18 ft circle centred 59 ft from the plate).
_MOUND_CENTRE_Y_FT = 59.0
_MOUND_RADIUS_FT = 9.0
_MOUND_TOP_RADIUS_FT = 2.5

# Why 1.07: SMPL's "head" joint is near the base of the skull, ~7% of standing height below the crown
# (roughly 0.93 H; anthropometric tables, e.g. Winter, Biomechanics of Human Movement), and the toe joint
# is ~on the ground.  Multiplying head-to-foot by 1.07 approximates crown height.
_HEAD_FACTOR = 1.07


def _smoothstep(t: np.ndarray) -> np.ndarray:
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3 - 2 * t)


def _smootherstep(t: np.ndarray) -> np.ndarray:
    """Perlin's quintic ease 6t^5 - 15t^4 + 10t^3: like smoothstep, but its acceleration is also zero at
    both ends, so an eased correction starts and stops without a jolt."""
    t = np.clip(t, 0.0, 1.0)
    return t ** 3 * (t * (6 * t - 15) + 10)


def _check(throws: str) -> None:
    if throws not in ("L", "R"):
        raise ValueError("throws must be 'L' or 'R'")


def lock_feet(p_ft: np.ndarray, fps: float, v_lo_ftps: float = 0.5, v_hi_ftps: float = 1.5) -> np.ndarray:
    """Hold each foot's horizontal position while it is planted, blending in/out (no hard switch).

    Plant weight = 1 below v_lo, 0 above v_hi, smoothstep between (a swinging foot moves >> 3 ft/s, a
    planted one ~0; the band absorbs lift noise).  Within each planted run the foot is pulled to its mean
    position by weight w(t); only the ankle and toe move (a cm-scale shift, leg proportions unaffected).
    The toe (ball of the foot) is the reference, not the ankle: the viewer pins the toe, and a planted foot
    that rolls onto its toes or sways (the rocker step's weight shift) moves the ankle while the toe stays.
    """
    out = p_ft.copy()
    for side in ("L", "R"):
        a = p_ft[:, _TOE[side], :2]
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

    p = lock_feet(p, fps)
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


def mound_z_ft(x_ft, y_ft):
    """Ground height (ft above home plate) at Statcast (x, y): the viewer's mound cone (web/src/field.js
    groundHeightFt), flat 10 in high out to 2.5 ft from its centre 59 ft from the plate, then falling
    linearly to 0 at 9 ft."""
    r = np.hypot(x_ft, np.asarray(y_ft) - _MOUND_CENTRE_Y_FT)
    return MOUND_Z_FT * np.clip((_MOUND_RADIUS_FT - r) / (_MOUND_RADIUS_FT - _MOUND_TOP_RADIUS_FT), 0.0, 1.0)


def segment_lengths(j: np.ndarray) -> np.ndarray:
    """(F, 23) length of every parent -> child bone (SMPL tree), in j's units."""
    return np.linalg.norm(j[:, 1:] - j[:, list(SMPL_PARENT[1:])], axis=-1)


def _two_bone(root, target, a, b, pole):
    """Per-frame two-bone IK (law of cosines): middle and end joints at exactly lengths a, b from root.

    The bend lies in the plane of root->target and `pole`.  An out-of-reach target is approached along
    the same line and stopped at full reach (0.9999 (a+b)); the caller keeps planted feet in reach."""
    u = target - root
    want = np.linalg.norm(u, axis=1)
    dist = np.clip(want, np.abs(a - b) + 1e-6, (a + b) * 0.9999)
    u = u / np.maximum(want, 1e-9)[:, None]
    cos_a = np.clip((a * a + dist * dist - b * b) / (2 * a * dist), -1.0, 1.0)
    w = pole - u * np.sum(pole * u, axis=1, keepdims=True)
    w = w / np.maximum(np.linalg.norm(w, axis=1), 1e-9)[:, None]
    mid = root + u * (a * cos_a)[:, None] + w * (a * np.sqrt(1 - cos_a ** 2))[:, None]
    return mid, root + u * dist[:, None]


def _envelope(req: np.ndarray, half_width: int = 6) -> np.ndarray:
    """A smooth curve >= req: a running max widens each requirement by `half_width` frames and a Gaussian
    with sigma = half_width / 3 rounds it off (+-3 sigma fits inside the widened plateau, so it keeps
    >= 99.7% of each peak); a final max covers the rest.  Why: the pelvis drop must never be less than
    needed (a planted foot would be out of reach) and must not kink either (a visible bob)."""
    d = gaussian_filter1d(maximum_filter1d(req, 2 * half_width + 1), half_width / 3, mode="nearest")
    return np.maximum(d, req)


def _soft_limit(x: np.ndarray, start: np.ndarray, limit: np.ndarray) -> np.ndarray:
    """x below `start` unchanged; above it, eased toward `limit` with tanh (value and slope continuous at
    `start`, never reaching `limit`).  Why: a hard clamp makes a joint stop dead (a knee snapping straight,
    a foot starting to slide at full speed); this one approaches the limit gradually."""
    span = np.maximum(limit - start, 1e-9)
    return np.where(x > start, start + span * np.tanh((x - start) / span), x)


def _hermite_ease(t: np.ndarray, t0: int, t1: int, t2: int, f1: float) -> np.ndarray:
    """0 before t0, f1 at t1, 1 from t2 on: cubic Hermite pieces with zero slope at t0 and t2 and the
    harmonic mean of the two secant slopes at t1 (Fritsch & Butland 1984), which keeps it monotone and C1
    -- a smoothstep generalised to pass through a chosen middle value."""
    s1, s2 = f1 / max(t1 - t0, 1), (1.0 - f1) / max(t2 - t1, 1)
    m1 = 2 * s1 * s2 / (s1 + s2) if s1 > 0 and s2 > 0 else 0.0
    out = np.zeros(len(t))
    for a, b, ya, yb, ma, mb in ((t0, t1, 0.0, f1, 0.0, m1), (t1, t2, f1, 1.0, m1, 0.0)):
        hgt = max(b - a, 1)
        u = np.clip((t - a) / hgt, 0.0, 1.0)
        val = ((2 * u ** 3 - 3 * u ** 2 + 1) * ya + (u ** 3 - 2 * u ** 2 + u) * hgt * ma
               + (-2 * u ** 3 + 3 * u ** 2) * yb + (u ** 3 - u ** 2) * hgt * mb)
        out = np.where((t >= a) & (t <= b), val, out)
    return np.where(t > t2, 1.0, out)


def _forward_only(py: np.ndarray, ll: int, ri: int) -> np.ndarray:
    """Change to add to the pelvis's y (ft) so that between frames `ll` and `ri` it never moves back.

    correct_to_mean calls it from leg lift to the last frame.  Why: from leg lift on the hips only travel
    toward the plate (the body's centre of mass speeds up toward home through the stride, e.g. Fleisig et
    al. 1999, and carries over the planted lead leg in the follow-through).  The averaged lift can instead drift BACK
    ~1 ft in the frames before a late foot strike (the pinned pivot toe drags the depth-compressed body
    back; ground_root's bridge only spans 0.1 s) and then rush forward, which snapped the back knee straight
    in 2 frames (Misiorowski, Sale); after release it stalled and restarted (Sale).  The y over [ll, ri] is replaced by its running minimum, smoothed with
    a Gaussian (sigma 3 frames = 0.05 s; smoothing a non-increasing curve keeps it non-increasing) so the
    flat spot neither starts nor stops with a kink.  Padding is extended linearly at each end's own slope (a
    Gaussian leaves a line unchanged), so the result meets the input at leg lift and release; the change is
    held, not dropped, outside [ll, ri].
    """
    out = np.zeros(len(py))
    c, pad = np.minimum.accumulate(py[ll:ri + 1]), 9
    if len(c) < 2:
        return out
    k = np.arange(1, pad + 1)
    q = np.concatenate([c[0] - (c[1] - c[0]) * k[::-1], c, c[-1] + (c[-1] - c[-2]) * k])
    out[ll:ri + 1] = gaussian_filter1d(q, 3.0)[pad:-pad] - py[ll:ri + 1]
    out[:ll], out[ri + 1:] = out[ll], out[ri]
    return out


def correct_to_mean(j_ft: np.ndarray, release_index: int, foot_strike_index: int, throws: str,
                    target: dict, leg_lift_index: int | None = None, knee_reach: float = 0.97,
                    info: dict | None = None, fps: float = 60.0, arm_ease_s: float = 0.45,
                    pivot_ease_s: float = 0.45) -> np.ndarray:
    """Bone-length-preserving correction so release_values match `target` (to 0.05 ft / 1 deg).

    Why not stretch or shift: the monocular lift squashes depth from the centre-field camera, so the stride
    comes out far too short (extension ~2.4 ft vs ~6.5).  Stretching y would stretch every bone, and
    shifting z would sink the feet into the mound.  Instead the delivery is rebuilt the way a longer stride
    would look, moving joints only rigidly or by IK at each frame's own bone lengths:

    1. Stride: the lead foot lands further toward the plate (-y), and sideways, by the upper-body shift
       below; its swing eases there over leg lift -> foot strike and it stays planted from foot strike
       on, toe on the mound surface (which also removes the lift's floating lead foot).  Moving the
       landing sideways too keeps the lead leg under the hips (otherwise it would be out of reach).
    2. Upper body (pelvis, hips, trunk, arms, head) moves forward for the extension gap and sideways for
       the release-x gap, eased over leg lift -> foot strike -> release (through the foot-strike pelvis
       position that needs the least drop, see body_ease) and held after.
    2b. Release height: trunk flexion about the pelvis (about +x, eased with the arm, see 5, held
       after) up to MAX_ADDED_FLEXION_DEG; any remaining gap lowers the upper body by knee flexion, eased
       over leg lift -> foot strike so that after foot strike the planted lead leg only has the pelvis
       travel over it, not a sinking pelvis as well (that folded the knee fast).
    3. Pelvis drop: wherever a planted foot (pivot until foot strike, lead after) -- or the lead foot late
       in its swing, weighted by its swing easing -- would be out of reach of a leg bent to `knee_reach`
       of its length (or the input's own bend, if straighter), the upper body is lowered a bit more,
       through a smooth envelope.  Both legs are then re-solved with two-bone IK at that frame's thigh /
       shin lengths, the knee bending in the input knee's own plane.  A swinging leg that still cannot
       reach is eased toward (never onto) 0.99 of its length, so the knee never snaps straight.
    4. Back foot: from foot strike it eases toward the body shift over `pivot_ease_s` (so it is still
       catching up through release, as a real back foot drags off the rubber); where the hips run away
       from it, it drags after them (toe height kept), the drag easing in over 0.1 s.
    4b. Forward only: from leg lift on, the pelvis never moves back toward the rubber (_forward_only).
    5. Arm angle: the throwing arm turns about the shoulder in the catcher's-view (x-z) plane (the plane
       release_values measures it in), eased over the `arm_ease_s` before release (or from foot strike, if
       that is earlier), held after (no snap back).
    Why the arm, trunk and back-foot easings have their own durations: the real lead foot plants only
    ~0.17 s before release; squeezing a 24 deg arm turn, 16 deg of trunk flexion and a ~3 ft back-foot move
    into those 10 frames folded the back knee ~35 deg/frame.  0.45 s is the span they had (and looked
    right) when foot strike was taken from the 3D lift.
    The amounts (arm rotation, x/y shift, flexion, lowering) come from fixed-point iteration on the
    release values; info["converged"] says whether it got within 0.005 ft / 0.05 deg.
    leg_lift_index: start of the stride easing (default fs - 2 (ri - fs), a typical leg-lift lead).
    fps: frame rate of j_ft (the easing durations are in seconds).
    info: optional out-dict with the applied rotation, flexion, shift, lowering, drop and iterations.
    """
    _check(throws)
    j0 = np.array(j_ft, float)
    n, ri, fs = len(j0), release_index, foot_strike_index
    ll = max(0, fs - 2 * (ri - fs)) if leg_lift_index is None else leg_lift_index
    t = np.arange(n)
    a0 = min(fs, ri - int(round(arm_ease_s * fps)))
    w_arm = _smootherstep((t - a0) / max(ri - a0, 1))
    w_piv = _smootherstep((t - fs) / max(pivot_ease_s * fps, 1.0))
    w_lead = _smootherstep((t - ll) / max(fs - ll, 1))
    # Back-foot drag easing: 6 frames = 0.1 s at 60 fps, about the time a real back foot takes to break
    # from the rubber and start sliding.
    w_drag = _smootherstep((t - fs) / (0.1 * fps))
    lead = "L" if throws == "R" else "R"
    sh, hand = _SHOULDER[throws], _HAND[throws]
    # (hip, knee, ankle, toe, frames the toe is planted, weight of its reach requirement)
    legs = [(_HIP[throws], _KNEE[throws], _ANKLE[throws], _TOE[throws], t <= fs, (t <= fs).astype(float)),
            (_HIP[lead], _KNEE[lead], _ANKLE[lead], _TOE[lead], t >= fs, np.where(t >= fs, 1.0, w_lead))]
    seg = lambda a, b: np.linalg.norm(j0[:, a] - j0[:, b], axis=1)  # noqa: E731
    # Reach each leg may use (ft): the input's own hip-ankle distance where nothing is changed yet (so the
    # untouched stance stays as it was, even a straight-legged one), blending over leg lift -> foot strike
    # to knee_reach of the leg -- one number used for the drop, the soft limit and the drag alike, so the
    # leg never meets a different limit from one frame to the next.
    leg_ft = {h: seg(h, k) + seg(k, a) for h, k, a, *_ in legs}
    reach_ft = {h: knee_reach * leg_ft[h] + (1 - w_lead) * np.maximum(seg(h, a) - knee_reach * leg_ft[h], 0.0)
                for h, k, a, *_ in legs}

    def need(hip_p, ank_p, leg_reach):
        """Pelvis drop (ft) that brings an ankle target into reach of a hip; past horizontal reach no drop
        is enough, so it grows steeply instead (a ranking penalty for choosing the foot-strike pelvis)."""
        h = np.linalg.norm((hip_p - ank_p)[..., :2], axis=-1)
        dz = hip_p[..., 2] - ank_p[..., 2]
        return np.where(h <= leg_reach, dz - np.sqrt(np.maximum(leg_reach ** 2 - h ** 2, 0.0)),
                        dz + 10.0 * (h - leg_reach))

    def body_ease(shift, land):
        """Upper-body easing 0 at leg lift -> f at foot strike -> 1 at release (held after), C1 and
        monotone.  f puts the pelvis at foot strike where the deeper of the two planted legs' required
        drops is smallest: a stride the lift never made has to be spanned by both legs, so neither the
        pivot (body running ahead) nor the lead leg (body lagging) is left out of reach."""
        # Lower bound: 0.1 below the stride's time fraction (fs - ll) / (ri - ll).  With the real, late
        # foot strike (10 frames before release) a pelvis far behind at foot strike has to sprint the rest
        # of the way in those 10 frames (seen on synthetic data: pelvis 3 -> 15 ft/s, knee 14 deg/frame).
        f = np.linspace(max((fs - ll) / max(ri - ll, 1) - 0.1, 0.0), 1.0, 51)[:, None]
        worst = np.zeros(len(f))
        for hip, knee, ank, toe, foot in ((_HIP[throws], _KNEE[throws], _ANKLE[throws], _TOE[throws], j0[0, _TOE[throws]]),
                                          (_HIP[lead], _KNEE[lead], _ANKLE[lead], _TOE[lead], land)):
            foot = foot.copy()
            foot[2] = mound_z_ft(foot[0], foot[1])
            r = reach_ft[hip][fs]
            worst = np.maximum(worst, need(j0[fs, hip] + f * shift * [1, 1, 0] + [0, 0, shift[2]], foot + j0[fs, ank] - j0[fs, toe], r))
        return _hermite_ease(t, ll, fs, ri, float(f[int(np.argmin(worst)), 0]))

    def build(alpha, theta, shift):
        j = j0.copy()
        # 2b. trunk flexion about +x through the pelvis (right-handed: +theta tips the head toward -y)
        c, s = np.cos(theta * w_arm)[:, None], np.sin(theta * w_arm)[:, None]
        r = j[:, _TRUNK] - j[:, None, 0]
        j[:, _TRUNK, 1] = j[:, None, 0, 1] + c * r[..., 1] - s * r[..., 2]
        j[:, _TRUNK, 2] = j[:, None, 0, 2] + s * r[..., 1] + c * r[..., 2]
        land = j0[fs, _TOE[lead]] + [shift[0], shift[1], 0.0]
        land[2] = mound_z_ft(land[0], land[1])
        w_body = body_ease(shift, land)
        # 5. arm angle: in-plane rotation of (sgn*dx, dz) about the shoulder, mirrored for L/R
        sgn = 1.0 if j[ri, hand, 0] - j[ri, sh, 0] >= 0 else -1.0
        c, s = np.cos(alpha * w_arm), np.sin(alpha * w_arm)
        for k in (_ELBOW[throws], _WRIST[throws], hand):
            r = j[:, k] - j[:, sh]
            u, w = sgn * r[:, 0], r[:, 2]
            j[:, k, 0] = j[:, sh, 0] + sgn * (c * u - s * w)
            j[:, k, 2] = j[:, sh, 2] + s * u + c * w
        # 2. upper body forward / sideways / down
        # (the lowering is done by foot strike: after it the planted lead leg only has the pelvis travel
        # over it, not also a sinking pelvis, which made the knee fold fast)
        j[:, _UPPER, :2] += w_body[:, None, None] * shift[:2]
        j[:, _UPPER, 1] += _forward_only(j[:, 0, 1], ll, n - 1)[:, None]
        j[:, _UPPER, 2] += (w_lead * shift[2])[:, None]
        # 1. toe targets (the foot, ankle -> toe, moves rigidly with its toe)
        toes = []
        for hip, knee, ank, toe, planted, _ in legs:
            p = j0[:, toe].copy()
            if toe == _TOE[lead]:
                p += w_lead[:, None] * (land - j0[fs, toe])
            else:  # pivot: from 0 on the rubber at foot strike toward the body's full shift, so the
                # lift's own back-leg pose relative to the hips comes back
                p[:, :2] += (w_body * w_piv)[:, None] * shift[:2]
            p[:, 2] = np.maximum(p[:, 2], mound_z_ft(p[:, 0], p[:, 1]))  # a swinging toe never digs in
            hold = p[int(np.argmax(planted))].copy()
            hold[2] = mound_z_ft(hold[0], hold[1])
            p[planted] = hold
            toes.append(p)
        # 3. pelvis drop: solve (hip_z - drop - ankle_z)^2 + horizontal^2 = reach^2 per foot
        req = np.zeros(n)
        for (hip, knee, ank, toe, planted, wt), p in zip(legs, toes):
            a_t = p + j0[:, ank] - j0[:, toe]
            req = np.maximum(req, wt * need(j[:, hip], a_t, reach_ft[hip]))
        drop = _envelope(np.maximum(req, 0.0))
        j[:, _UPPER, 2] -= drop[:, None]
        for (hip, knee, ank, toe, planted, _), p in zip(legs, toes):
            a_t = p + j0[:, ank] - j0[:, toe]
            leg, reach = leg_ft[hip], reach_ft[hip]
            d = a_t - j[:, hip]
            if toe == _TOE[throws]:
                # 4. drag: a back foot the hips run away from slides after them (toe height kept), softly
                # limited to the horizontal distance the leg's reach allows
                h = np.linalg.norm(d[:, :2], axis=1)
                h_max = np.sqrt(np.maximum(reach ** 2 - d[:, 2] ** 2, 0.0))
                h_soft = _soft_limit(h, np.maximum(h_max - 0.08 * leg, 0.0), h_max)
                h_eff = h - w_drag * (t > fs) * (h - h_soft)
                d[:, :2] *= (h_eff / np.maximum(h, 1e-9))[:, None]
            # swinging legs: ease the reach toward 0.99 leg instead of a hard stop at full length
            want = np.linalg.norm(d, axis=1)
            soft = _soft_limit(want, reach, np.maximum(0.99 * leg, reach + 1e-6))
            d *= np.where(planted, 1.0, soft / np.maximum(want, 1e-9))[:, None]
            # Knee bend direction: the input knee's offset from the INPUT hip->ankle line, plus a 0.1 ft
            # forward (-y) bias.  Taking (knee - hip) raw would flip the plane when the input leg is nearly
            # straight and the corrected leg points elsewhere; the bias keeps it stable then.
            u0 = j0[:, ank] - j0[:, hip]
            u0 /= np.linalg.norm(u0, axis=1, keepdims=True)
            k0 = j0[:, knee] - j0[:, hip]
            pole = k0 - u0 * np.sum(k0 * u0, axis=1, keepdims=True) + [0.0, -0.1, 0.0]
            j[:, knee], j[:, ank] = _two_bone(j[:, hip], j[:, hip] + d, seg(hip, knee), seg(knee, ank), pole)
            j[:, toe] = j[:, ank] + j0[:, toe] - j0[:, ank]
        return j, drop, w_body

    alpha = theta = 0.0
    shift = np.zeros(3)
    cap = np.radians(MAX_ADDED_FLEXION_DEG)
    want_y = RUBBER_Y_FT - target["extension_ft"]
    for it in range(100):
        j, drop, w_body = build(alpha, theta, shift)
        rv = release_values(j, ri, throws)
        gaps = np.array([target["release_x_ft"] - rv["release_x_ft"], want_y - j[ri, hand, 1],
                         target["release_z_ft"] - rv["release_z_ft"]])
        g_a = target["arm_angle_deg"] - rv["arm_angle_deg"]
        if np.abs(gaps).max() < 0.005 and abs(g_a) < 0.05:
            break
        alpha += np.radians(g_a)
        shift[:2] += gaps[:2]
        # Height: Newton step on the flexion first (about +x, d(hand z)/d(theta) is the hand's current y
        # offset from the pelvis), clipped to [0, cap]; whatever the clip leaves goes to the lowering.
        dzdt = j[ri, hand, 1] - j[ri, 0, 1]
        new = float(np.clip(theta + gaps[2] / dzdt, 0.0, cap)) if abs(dzdt) > 0.1 else theta
        shift[2] += gaps[2] - (new - theta) * dzdt
        theta = new
    if info is not None:
        info.update(arm_rotation_deg=float(np.degrees(alpha)), trunk_flexion_deg=float(np.degrees(theta)),
                    upper_body_shift_ft=[float(v) for v in shift],
                    body_fraction_at_foot_strike=float(w_body[fs]), max_pelvis_drop_ft=float(drop.max()),
                    pelvis_drop_at_release_ft=float(drop[ri]), iterations=it + 1,
                    converged=bool(np.abs(gaps).max() < 0.005 and abs(g_a) < 0.05))
    return j


def smooth_extra(j_ft: np.ndarray, fps: float, until: int, throws: str, sigma_s: float = 0.12,
                 blend_s: float = 0.5) -> np.ndarray:
    """Zero-phase extra smoothing of the windup, before the leg-lift peak `until` only (opt-in, per pitcher).

    Why: Cease's lead foot/knee still show a ~150 ft/s^2 acceleration spike and a creeping foot before the
    leg lift that phase_smooth (3-frame sigma) leaves, because the averaged lifts disagree there.  Joint
    positions get a wide Gaussian (sigma_s, symmetric = no time shift) blended in with weight 1 until
    `until` - blend_s, easing (smootherstep: C2) to 0 at `until`, so the leg-lift peak and everything after
    it are untouched bit for bit.  Then the skeleton is rebuilt so nothing else changes: bones keep their
    exact per-frame lengths (child = parent + length * smoothed direction, root first), and the planted
    pivot foot (ankle and toe) stays where it was, its knee re-solved by two-bone IK from the moved hip.
    """
    _check(throws)
    j = np.asarray(j_ft, float)
    sm = gaussian_filter1d(j, sigma_s * fps, axis=0, mode="nearest")
    u = np.clip((until - np.arange(len(j))) / max(blend_s * fps, 1.0), 0.0, 1.0)
    w = _smootherstep(u)[:, None, None]
    tgt = j + w * (sm - j)
    out = j.copy()
    out[:, 0] = tgt[:, 0]
    for k in range(1, j.shape[1]):  # SMPL_PARENT lists parents before children
        p = SMPL_PARENT[k]
        d = tgt[:, k] - tgt[:, p]
        d /= np.maximum(np.linalg.norm(d, axis=1), 1e-9)[:, None]
        out[:, k] = out[:, p] + np.linalg.norm(j[:, k] - j[:, p], axis=1)[:, None] * d
    # the pivot (throwing-side) leg is the planted one in the windup
    hip, knee, ank, toe = _HIP[throws], _KNEE[throws], _ANKLE[throws], _TOE[throws]
    out[:, ank], out[:, toe] = j[:, ank], j[:, toe]
    a = np.linalg.norm(j[:, knee] - j[:, hip], axis=1)
    b = np.linalg.norm(j[:, ank] - j[:, knee], axis=1)
    pole = j[:, knee] - j[:, hip]
    out[:, knee], _ = _two_bone(out[:, hip], j[:, ank], a, b, pole)
    out[w[:, 0, 0] == 0] = j[w[:, 0, 0] == 0]  # weight 0: exactly the input (no float residue from the rebuild)
    return out


def smooth_knee(j_ft: np.ndarray, fps: float, start: int, until: int, throws: str) -> np.ndarray:
    """Remove the lead-knee "buckle" before the leg-lift peak `until` (opt-in, per pitcher).

    Why: the lead knee's flexion rate jumps from ~1.6 to ~3.3 deg/frame over the ~4 frames before the peak
    (angular accel spike ~ -1800 deg/s^2, then a reversal), which reads as the knee giving way.  The hip-knee-ankle
    angle on [start, until] is replaced by a cubic Hermite that matches the data's value and slope at both ends
    (C1 join; a quintic that also matched curvature overshoots, because the rate must still reach the data's
    value at `until`), so frames >= `until` stay bit-identical.  Only the angle changes: hip and knee stay put
    and the shank + foot + toe rotate rigidly about the knee axis (normal of the hip-knee-ankle plane), so
    every bone length is exact and the thigh (leg-lift height) is untouched.  Zero-phase: no time shift.
    """
    _check(throws)
    j = np.asarray(j_ft, float).copy()
    lead = "L" if throws == "R" else "R"
    hip, knee, ank, toe = _HIP[lead], _KNEE[lead], _ANKLE[lead], _TOE[lead]
    u, v = j[:, hip] - j[:, knee], j[:, ank] - j[:, knee]
    uh, vh = u / np.linalg.norm(u, axis=1)[:, None], v / np.linalg.norm(v, axis=1)[:, None]
    ang = np.arccos(np.clip((uh * vh).sum(1), -1, 1))
    m = lambda i: (ang[i + 1] - ang[i - 1]) / 2  # slope per frame, central difference of the original data
    T = until - start
    s = np.arange(T + 1) / T
    new = ((2 * s**3 - 3 * s**2 + 1) * ang[start] + (s**3 - 2 * s**2 + s) * T * m(start)
           + (-2 * s**3 + 3 * s**2) * ang[until] + (s**3 - s**2) * T * m(until))
    delta = np.zeros(len(j))
    delta[start:until + 1] = new - ang[start:until + 1]  # new angle - old angle; > 0 opens the knee
    n = np.cross(uh, vh)
    n /= np.maximum(np.linalg.norm(n, axis=1), 1e-9)[:, None]
    th = delta[:, None]
    kn = j[:, knee]
    for q in (ank, toe):  # Rodrigues rotation by +delta about n = u x v: turns v away from u, so the knee angle grows by delta
        r = j[:, q] - kn
        rot = r * np.cos(th) + np.cross(n, r) * np.sin(th) + n * (n * r).sum(1)[:, None] * (1 - np.cos(th))
        j[:, q] = kn + rot
    src = np.asarray(j_ft, float)
    j[:start], j[until:] = src[:start], src[until:]  # exact, no float residue outside the window
    return j
