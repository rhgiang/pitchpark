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
# Joints that move with the pelvis (pelvis, hips, spine, neck, head, both arms), and the part of those
# that bends forward with the trunk (everything above the pelvis except the hips).
_UPPER = [0, 1, 2, 3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
_TRUNK = [3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
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


def _check(throws: str) -> None:
    if throws not in ("L", "R"):
        raise ValueError("throws must be 'L' or 'R'")


def lock_feet(p_ft: np.ndarray, fps: float, v_lo_ftps: float = 0.5, v_hi_ftps: float = 1.5) -> np.ndarray:
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


def correct_to_mean(j_ft: np.ndarray, release_index: int, foot_strike_index: int, throws: str,
                    target: dict, leg_lift_index: int | None = None, knee_reach: float = 0.97,
                    info: dict | None = None) -> np.ndarray:
    """Bone-length-preserving correction so release_values match `target` (to 0.05 ft / 1 deg).

    Why not stretch or shift: the monocular lift squashes depth from the centre-field camera, so stride
    and forward lean come out far too short (extension ~2.4 ft vs ~6.5).  Stretching y would stretch every
    bone, and shifting z would sink the feet into the mound.  Instead the delivery is rebuilt the way a
    longer stride would look, moving joints only rigidly or by IK at each frame's own bone lengths:

    1. Stride: the lead foot lands further toward the plate (-y), and sideways, by the upper-body shift
       below; its swing eases there over leg lift -> foot strike and it stays planted from foot strike
       on, toe on the mound surface (which also removes the lift's floating lead foot).  Moving the
       landing sideways too keeps the lead leg under the hips (otherwise it would be out of reach).
    2. Upper body (pelvis, hips, trunk, arms, head) moves forward for the extension gap and sideways for
       the release-x gap, eased over leg lift -> release, held after.  The pivot foot, once off the
       rubber (after foot strike), catches up with that shift by release; a toe never goes under the mound.
    3. Pelvis drop: wherever a planted foot (pivot until foot strike, lead after) would be out of reach of
       a leg bent at least to `knee_reach` of its length (or the input's own bend, if straighter), the
       upper body is lowered just enough, through a smooth envelope.  Both legs are then re-solved with
       two-bone IK at that frame's thigh/shin lengths, the knee bending in the input knee's own plane.
    4. Trunk flexion about the pelvis (about +x, eased foot strike -> release, held after) takes up the
       remaining release-height gap: the forward lean the camera also squashed.
    5. Arm angle: the throwing arm turns about the shoulder in the catcher's-view (x-z) plane (the plane
       release_values measures it in), eased over foot strike -> release, held after (no snap back).
    The five amounts are found by fixed-point iteration on the release values.
    leg_lift_index: start of the stride easing (default fs - 2 (ri - fs), a typical leg-lift lead).
    info: optional out-dict with the applied rotation, flexion, shift, stride, drop and iterations.
    """
    _check(throws)
    j0 = np.array(j_ft, float)
    n, ri, fs = len(j0), release_index, foot_strike_index
    ll = max(0, fs - 2 * (ri - fs)) if leg_lift_index is None else leg_lift_index
    t = np.arange(n)
    w_arm = _smoothstep((t - fs) / max(ri - fs, 1))
    w_body = _smoothstep((t - ll) / max(ri - ll, 1))
    w_lead = _smoothstep((t - ll) / max(fs - ll, 1))
    lead = "L" if throws == "R" else "R"
    sh, hand = _SHOULDER[throws], _HAND[throws]
    # (hip, knee, ankle, toe, frames the toe is planted): pivot until foot strike, lead from foot strike
    legs = [(_HIP[s], _KNEE[s], _ANKLE[s], _TOE[s], m) for s, m in ((throws, t <= fs), (lead, t >= fs))]
    seg = lambda a, b: np.linalg.norm(j0[:, a] - j0[:, b], axis=1)  # noqa: E731

    def build(alpha, theta, shift):
        j = j0.copy()
        # 4. trunk flexion about +x through the pelvis (right-handed: +theta tips the head toward -y)
        c, s = np.cos(theta * w_arm)[:, None], np.sin(theta * w_arm)[:, None]
        r = j[:, _TRUNK] - j[:, None, 0]
        j[:, _TRUNK, 1] = j[:, None, 0, 1] + c * r[..., 1] - s * r[..., 2]
        j[:, _TRUNK, 2] = j[:, None, 0, 2] + s * r[..., 1] + c * r[..., 2]
        # 5. arm angle: in-plane rotation of (sgn*dx, dz) about the shoulder, mirrored for L/R
        sgn = 1.0 if j[ri, hand, 0] - j[ri, sh, 0] >= 0 else -1.0
        c, s = np.cos(alpha * w_arm), np.sin(alpha * w_arm)
        for k in (_ELBOW[throws], _WRIST[throws], hand):
            r = j[:, k] - j[:, sh]
            u, w = sgn * r[:, 0], r[:, 2]
            j[:, k, 0] = j[:, sh, 0] + sgn * (c * u - s * w)
            j[:, k, 2] = j[:, sh, 2] + s * u + c * w
        # 2. upper body forward / sideways
        j[:, _UPPER] += w_body[:, None, None] * shift
        # 1. toe targets (the foot, ankle -> toe, moves rigidly with its toe)
        toes = []
        for hip, knee, ank, toe, planted in legs:
            p = j0[:, toe].copy()
            if toe == _TOE[lead]:
                land = j0[fs, toe] + [shift[0], shift[1], 0.0]
                land[2] = mound_z_ft(land[0], land[1])
                p += w_lead[:, None] * (land - j0[fs, toe])
            else:  # pivot: from 0 on the rubber at foot strike to the body's full shift by release, so
                # the lift's own back-leg pose relative to the hips comes back (it would hang straight
                # at full reach if it lagged the body)
                p[:, :2] += (w_body * w_arm)[:, None] * shift[:2]
            p[:, 2] = np.maximum(p[:, 2], mound_z_ft(p[:, 0], p[:, 1]))  # a swinging toe never digs in
            hold = p[int(np.argmax(planted))].copy()
            hold[2] = mound_z_ft(hold[0], hold[1])
            p[planted] = hold
            toes.append(p)
        # 3. pelvis drop: solve (hip_z - drop - ankle_z)^2 + horizontal^2 = reach^2 for each planted foot
        req = np.zeros(n)
        for (hip, knee, ank, toe, planted), p in zip(legs, toes):
            a_t = p + j0[:, ank] - j0[:, toe]
            reach = np.maximum(knee_reach * (seg(hip, knee) + seg(knee, ank)), seg(hip, ank))
            h = np.linalg.norm((j[:, hip] - a_t)[:, :2], axis=1)
            need = j[:, hip, 2] - a_t[:, 2] - np.sqrt(np.maximum(reach ** 2 - h ** 2, 0.0))
            req = np.maximum(req, np.where(planted, need, 0.0))
        drop = _envelope(req)
        j[:, _UPPER, 2] -= drop[:, None]
        for (hip, knee, ank, toe, planted), p in zip(legs, toes):
            a_t = p + j0[:, ank] - j0[:, toe]
            if toe == _TOE[throws]:
                # Drag: once off the rubber, a back foot the hips have run away from slides after them
                # (toe height kept) just far enough to stay within knee_reach, as a real back foot drags,
                # instead of the leg locking straight with the foot hanging off its target.
                d = a_t - j[:, hip]
                r = knee_reach * (seg(hip, knee) + seg(knee, ank))
                h = np.linalg.norm(d[:, :2], axis=1)
                h_max = np.sqrt(np.maximum(r ** 2 - d[:, 2] ** 2, 0.0))
                pull = (t > fs) & (h > h_max)
                d[pull, :2] *= (h_max[pull] / h[pull])[:, None]
                a_t = j[:, hip] + d
            pole = j0[:, knee] - j0[:, hip] + [0.0, -1e-3, 0.0]  # input knee's own plane (+ tiny forward bias)
            j[:, knee], j[:, ank] = _two_bone(j[:, hip], a_t, seg(hip, knee), seg(knee, ank), pole)
            j[:, toe] = j[:, ank] + j0[:, toe] - j0[:, ank]
        return j, drop

    alpha = theta = 0.0
    shift = np.zeros(3)
    want_y = RUBBER_Y_FT - target["extension_ft"]
    for it in range(100):
        j, drop = build(alpha, theta, shift)
        rv = release_values(j, ri, throws)
        g_a = target["arm_angle_deg"] - rv["arm_angle_deg"]
        g_x = target["release_x_ft"] - rv["release_x_ft"]
        g_y = want_y - j[ri, hand, 1]
        g_z = target["release_z_ft"] - rv["release_z_ft"]
        if max(abs(g_x), abs(g_y), abs(g_z)) < 0.005 and abs(g_a) < 0.05:
            break
        alpha += np.radians(g_a)
        shift[:2] += [g_x, g_y]
        # Newton step: rotating about +x, d(hand z)/d(theta) is the hand's current y offset from the pelvis
        dzdt = j[ri, hand, 1] - j[ri, 0, 1]
        if abs(dzdt) > 0.1:
            theta += float(np.clip(g_z / dzdt, -0.2, 0.2))
    if info is not None:
        info.update(arm_rotation_deg=float(np.degrees(alpha)), trunk_flexion_deg=float(np.degrees(theta)),
                    upper_body_shift_ft=[float(v) for v in shift], max_pelvis_drop_ft=float(drop.max()),
                    pelvis_drop_at_release_ft=float(drop[ri]), iterations=it + 1,
                    converged=bool(max(abs(g_x), abs(g_y), abs(g_z)) < 0.005 and abs(g_a) < 0.05))
    return j
