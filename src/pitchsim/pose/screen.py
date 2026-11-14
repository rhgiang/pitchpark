"""Screening: is a clip's 2D pose usable?  Pure numpy, thresholds are first guesses to tune on data."""
import warnings

import numpy as np

from pitchsim.pose.pose2d import Pose2D
from pitchsim.pose.release import wrist_idx

MIN_SECONDS = 1.0      # a windup+release is ~1.5 s; shorter is a cropped fragment
MIN_PERSON_FRAC = 0.7  # fraction of frames with a confidently detected body
CONF = 0.3
MIN_WRIST_RISE = 0.45  # torso lengths the throwing wrist must rise above the shoulder


def _boxes(p: Pose2D):
    """Per-frame bbox (x0,y0,x1,y1) of confident keypoints; NaN rows where none are confident."""
    m = p.conf > CONF
    x = np.where(m, p.xy[..., 0], np.nan)
    y = np.where(m, p.xy[..., 1], np.nan)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")  # all-NaN frames are expected (no person)
        have = m.sum(1) >= 5
        b = np.stack([np.nanmin(x, 1), np.nanmin(y, 1), np.nanmax(x, 1), np.nanmax(y, 1)], 1)
    b[~have] = np.nan
    return b, have


def screen(p: Pose2D, throws: str) -> tuple[bool, str]:
    """(ok, reason); reason is '' when ok."""
    w, h = p.size
    if len(p.xy) < MIN_SECONDS * p.fps:   # why: too short to contain a full delivery
        return False, "short"
    b, have = _boxes(p)
    if have.mean() < MIN_PERSON_FRAC:     # why: detector lost the pitcher (cut to another view/crowd)
        return False, "no_pitcher"
    cx = (b[:, 0] + b[:, 2]) / 2
    cx = cx[have]
    if np.any(np.abs(np.diff(cx)) > 0.5 * w):  # why: a person can't move half the frame in one frame; it's a cut
        return False, "camera_cut"
    height = np.nanmedian(b[:, 3] - b[:, 1])
    # why: centre-field view = pitcher large and near the middle; other angles put him at an edge or tiny
    if not (0.25 * w < np.median(cx) < 0.75 * w) or height < 0.15 * h:
        return False, "not_center_field"
    # why: the tracker often sits on the catcher/batter when Ohtani is off-frame (low-home angle).
    # A real delivery lifts the throwing wrist well above the shoulder; those people never do.
    # 0.45 torso lengths splits the 13 clips I checked by eye (good >= 0.50, wrong person <= 0.41).
    wr, sh, hip = wrist_idx(throws), wrist_idx(throws) - 4, wrist_idx(throws) + 2
    seen = (p.conf[:, [wr, sh, hip]] > CONF).all(1)
    torso = np.abs(p.xy[:, sh, 1] - p.xy[:, hip, 1])
    seen &= torso > 10
    if not seen.any() or ((p.xy[:, sh, 1] - p.xy[:, wr, 1])[seen] / torso[seen]).max() < MIN_WRIST_RISE:
        return False, "no_pitcher"
    arm = [wrist_idx(throws) - 4, wrist_idx(throws) - 2, wrist_idx(throws)]  # shoulder, elbow, wrist
    if np.median(p.conf[:, arm]) < 0.4:   # why: release-frame detection and later lifting rely on the arm
        return False, "low_confidence"
    return True, ""
