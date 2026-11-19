"""Reshape the lead knee's extend-then-re-bend into the foot plant (opt-in, per pitcher).

Why: before foot strike Cease's lead knee opens to ~147 deg then re-bends to ~46 deg in ~10 frames, the
largest knee acceleration before release (the leg is pulled into place).  Plain Gaussian smoothing only
blunts the extension peak.  Here the hip-knee-ankle angle on [start, until] is replaced by a jerk-penalised
smoothing spline of the data (minimise mu*|new - old|^2 + |third difference of new|^2) whose first and last
3 frames are pinned to the data, so value, velocity and acceleration join the data at both ends (discrete C2)
and frames outside the window stay bit-identical.  A larger `lam` spreads the fall over more frames.  As in
smooth_knee, only the angle changes: hip and knee stay put and shank + foot + toe rotate rigidly about the knee
axis, so every bone length is exact.  Zero-phase: no time shift.
"""
import numpy as np

from pitchsim.pose.anchor import _ANKLE, _HIP, _KNEE, _TOE, _check


def knee_angle_deg(j: np.ndarray, lead: str) -> np.ndarray:
    u, v = j[:, _HIP[lead]] - j[:, _KNEE[lead]], j[:, _ANKLE[lead]] - j[:, _KNEE[lead]]
    c = (u * v).sum(1) / np.linalg.norm(u, axis=1) / np.linalg.norm(v, axis=1)
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def reshape_stride(j_ft: np.ndarray, start: int, until: int, throws: str, lam: float = 1.0,
                   fall_from: int = 10**9, fall_weight: float = 1.0) -> np.ndarray:
    _check(throws)
    src = np.asarray(j_ft, float)
    j = src.copy()
    lead = "L" if throws == "R" else "R"
    knee, ank, toe = _KNEE[lead], _ANKLE[lead], _TOE[lead]
    u, v = src[:, _HIP[lead]] - src[:, knee], src[:, ank] - src[:, knee]
    uh, vh = u / np.linalg.norm(u, axis=1)[:, None], v / np.linalg.norm(v, axis=1)[:, None]
    old = np.arccos(np.clip((uh * vh).sum(1), -1, 1))
    n_ = until - start + 1
    D = np.diff(np.eye(n_), 3, axis=0)
    free = np.arange(3, n_ - 3)
    y0 = old[start:until + 1]
    wd = np.where(np.arange(start, until + 1) >= fall_from, np.sqrt(fall_weight), 1.0)  # data weight: low on the fall = free to spread
    # unknowns: new angles at free frames; pinned frames keep the data
    A = np.vstack([D[:, free] * np.sqrt(lam), np.diag(wd[free])])
    pinned = np.ones(n_, bool); pinned[free] = False
    b = np.concatenate([-(D[:, pinned] @ y0[pinned]) * np.sqrt(lam), y0[free] * wd[free]])
    new = y0.copy()
    new[free] = np.linalg.lstsq(A, b, rcond=None)[0]
    delta = np.zeros(len(j)); delta[start:until + 1] = new - y0
    n = np.cross(uh, vh)
    n /= np.maximum(np.linalg.norm(n, axis=1), 1e-9)[:, None]
    th, kn = delta[:, None], src[:, knee]
    for q in (ank, toe):  # Rodrigues rotation by +delta about u x v (same as anchor.smooth_knee)
        r = src[:, q] - kn
        j[:, q] = kn + r * np.cos(th) + np.cross(n, r) * np.sin(th) + n * (n * r).sum(1)[:, None] * (1 - np.cos(th))
    j[:start], j[until:] = src[:start], src[until:]
    return j

