"""Release frame from 2D pose: the frame where the throwing wrist is fastest."""
import numpy as np
from scipy.signal import find_peaks, medfilt

from pitchsim.pose.pose2d import Pose2D


def wrist_idx(throws: str) -> int:
    """COCO-17 wrist index of the throwing arm (9 left, 10 right)."""
    return 10 if throws == "R" else 9


def lead_ankle_idx(throws: str) -> int:
    """COCO-17 ankle of the lead (glove-side) leg: 15 left for a RHP, 16 right for a LHP."""
    return 15 if throws == "R" else 16


def release_frame(p: Pose2D, throws: str) -> int:
    """Frame of the first major peak of throwing-wrist speed.

    Why: the hand is at its fastest right as the ball leaves it (whip-like arm chain); after
    release it decelerates.  Speed is taken on a 5-frame median of position so one-frame
    keypoint jitter doesn't win the argmax.  Low-confidence frames are masked out.
    ponytail: peak speed can precede true release by a few frames; the hand labels measure that.
    """
    j = wrist_idx(throws)
    # Median (not mean) filter: tracker glitches are one-to-few-frame jumps to another person's
    # hand, and a median removes them where a moving average would smear them into a fake peak.
    sm = np.stack([medfilt(p.xy[:, j, d], 5) for d in (0, 1)], 1)
    speed = np.linalg.norm(np.gradient(sm, axis=0), axis=1)
    speed[p.conf[:, j] < 0.3] = 0
    # Real wrist peaks in these clips are ~40-110 px/frame; faster than 8% of frame width per
    # frame is a tracking jump or camera cut, so it cannot be the release.
    speed[speed > 0.08 * p.size[0]] = 0
    speed[:3] = speed[-3:] = 0   # filter edge effects
    # Why the FIRST big peak, not the global max: the pitch is the first large throwing-arm action
    # in a pitch clip; later peaks are fielding plays or return throws, which can be faster.
    # 0.5 x max keeps real release peaks while skipping small early wobble; distance=15 frames
    # (0.25 s) merges the double bumps one swing can produce.
    peaks, _ = find_peaks(speed, height=0.5 * speed.max(), distance=15)
    return int(peaks[0]) if len(peaks) else int(np.argmax(speed))


def foot_strike_frame(p: Pose2D, throws: str, release: int, tol_torso: float = 0.13) -> int | None:
    """Frame the lead (glove-side) foot plants, from the 2D lead ankle; None if it never settles.

    Why 2D and not the 3D lift: from the centre-field camera the stride runs along the camera axis, and
    the monocular lift compresses that depth, so its 3D lead ankle slows ~0.45 s before release while the
    real foot is still reaching forward just above the dirt.  In the image the foot keeps moving (down and
    sideways) until it plants, so "the ankle reaches its planted spot and stays there" is visible in 2D.
    Planted spot = median ankle position over the last 3 frames up to release (the lead foot is down at
    release).  Foot strike = first frame, within release - 0.6 s .. release - 0.05 s (the window starts after
    the leg-lift peak, which is ~0.9 s before release; the search takes the highest ankle inside it), from which the ankle stays within `tol_torso` x torso length
    (shoulder-mid to hip-mid, px) of that spot through release.  Both x and y count: the ankle reaches
    ground height in the image while the foot is still sliding toward the plate.  Position is 5-frame
    median filtered (one- or two-frame tracker swaps onto the other foot are common at landing).
    tol_torso = 0.13 was picked against 12 hand labels (in-sample; it lands ~2-3 frames before the
    "foot flat" label, i.e. near heel contact).  Biomechanics: foot contact -> release is ~0.15-0.2 s in
    pro pitchers (Fleisig et al. 1999), which bounds the window.
    """
    k = lead_ankle_idx(throws)
    a = np.stack([medfilt(p.xy[:, k, d], 5) for d in (0, 1)], 1)
    w = slice(max(release - 30, 0), release)
    torso = np.median(np.linalg.norm(p.xy[w, 5:7].mean(1) - p.xy[w, 11:13].mean(1), axis=1))
    near = np.linalg.norm(a - np.median(a[release - 2: release + 1], axis=0), axis=1) < tol_torso * torso
    lo, hi = max(release - int(round(0.6 * p.fps)), 0), release - int(round(0.05 * p.fps))
    lo = lo + int(np.argmin(a[lo:release, 1]))  # leg-lift peak: image y grows downward
    for f in range(lo, hi + 1):
        if near[f: release + 1].all():
            return f
    return None
