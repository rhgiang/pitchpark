"""Opt-in smoothing of King's finished delivery (post-process of the shipped JSON; imports anchor, never edits it).

Two zero-phase passes, each blended back to the untouched data with a C1/C2 taper:

1. Windup (before the leg-lift peak): `anchor.smooth_extra` with a window that reaches a few frames past the peak.
   King's lead knee/ankle lurch (speed 4.7 -> 9 -> 4 ft/s) and the head pulses twice in the ~10 frames around the
   peak, because only 2 train clips are averaged there.  The peak frame itself keeps weight ~0, so the lift's
   height and timing stay.
2. Pre-release (last `pre_s` before the release frame): the upper body (spine, neck, head, collars, shoulders,
   arms, hands) is smoothed with a small Gaussian.  The pelvis, hips and both legs are not touched, so the planted
   feet stay bit-identical; the weight tapers to 0 at `release - hold` so the release frame (and the `hold`
   frames before it) keep the exact pose, hand position included.  Bones are rebuilt root-first at their exact
   per-frame lengths (child = parent + length * smoothed direction), as in `anchor.smooth_extra`.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d

from pitchsim.pose.anchor import SMPL_PARENT, _smootherstep, smooth_extra

# everything above the hips: sp1 sp2 sp3 neck collars head shoulders elbows wrists hands
UPPER = (3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23)


def smooth_pre_release(j_ft: np.ndarray, fps: float, release: int, throws: str, sigma_s: float = 0.025,
                       span_s: float = 0.4, hold: int = 1, blend_s: float = 0.1) -> np.ndarray:
    """Light Gaussian smoothing of the upper body over [release - span, release - hold], tapered to 0 at both ends.

    Weight is 0 for frames >= release - hold (release pose and hand position exact) and 0 at release - span,
    easing in over `blend_s` with smootherstep (C2), so the seams carry no step in position or velocity.
    """
    j = np.asarray(j_ft, float)
    n = len(j)
    t = np.arange(n)
    b = max(blend_s * fps, 1.0)
    up = _smootherstep(np.clip((t - (release - span_s * fps)) / b, 0, 1))
    down = _smootherstep(np.clip(((release - hold) - t) / b, 0, 1))
    w = (up * down)[:, None, None]
    sm = gaussian_filter1d(j, sigma_s * fps, axis=0, mode="nearest")
    tgt = j + w * (sm - j)
    out = j.copy()
    for k in range(1, j.shape[1]):
        if k not in UPPER:
            continue
        p = SMPL_PARENT[k]
        d = tgt[:, k] - tgt[:, p] if p in UPPER else tgt[:, k] - j[:, p]
        d /= np.maximum(np.linalg.norm(d, axis=1), 1e-9)[:, None]
        out[:, k] = out[:, p] + np.linalg.norm(j[:, k] - j[:, p], axis=1)[:, None] * d
    free = w[:, 0, 0] == 0
    out[free] = j[free]  # weight 0: exactly the input
    return out


def smooth_king(j_ft: np.ndarray, fps: float, leg_lift: int, release: int, throws: str, windup_after: int = 4,
                windup_sigma_s: float = 0.12, windup_blend_s: float = 0.3, pre_sigma_s: float = 0.03,
                pre_span_s: float = 0.4) -> np.ndarray:
    """Windup pass, then pre-release pass (they do not overlap: the first ends well before release - span)."""
    o = smooth_extra(j_ft, fps, leg_lift + windup_after, throws, windup_sigma_s, windup_blend_s)
    return smooth_pre_release(o, fps, release, throws, pre_sigma_s, pre_span_s)
