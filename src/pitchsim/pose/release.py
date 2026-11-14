"""Release frame from 2D pose: the frame where the throwing wrist is fastest."""
import numpy as np
from scipy.signal import medfilt

from pitchsim.pose.pose2d import Pose2D


def wrist_idx(throws: str) -> int:
    """COCO-17 wrist index of the throwing arm (9 left, 10 right)."""
    return 10 if throws == "R" else 9


def release_frame(p: Pose2D, throws: str) -> int:
    """Frame of peak throwing-wrist speed.

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
    return int(np.argmax(speed))
