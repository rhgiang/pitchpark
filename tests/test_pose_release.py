"""Release-frame and screening tests on SYNTHETIC Pose2D (no real video)."""
import numpy as np

from pitchsim.pose.pose2d import Pose2D
from pitchsim.pose.release import release_frame
from pitchsim.pose.screen import screen

W, H, FPS = 1280, 720, 60.0


def synth(n=120, wrist=10, peak=40):
    """Synthetic: standing figure at frame centre; chosen wrist speed is a Gaussian peaking at `peak`."""
    xy = np.zeros((n, 17, 2))
    xy[:] = [W / 2, H / 2]
    xy[:, :, 1] += np.linspace(-150, 150, 17)       # spread keypoints vertically: 300 px tall
    speed = 2 + 40 * np.exp(-0.5 * ((np.arange(n) - peak) / 5.0) ** 2)
    xy[:, wrist, 0] += np.cumsum(speed)              # position = integral of speed
    xy[:, wrist, 1] -= 150 * np.exp(-0.5 * ((np.arange(n) - peak) / 8.0) ** 2)  # arm swings overhead
    return Pose2D(xy=xy, conf=np.full((n, 17), 0.9), fps=FPS, size=(W, H))


def test_release_frame_peak():
    assert abs(release_frame(synth(), "R") - 40) <= 1


def test_release_frame_lefty():
    p = synth(wrist=9)
    p.xy[:, 10, 0] += np.cumsum(np.where(np.arange(120) == 80, 60, 1))  # right wrist: decoy spike at 80
    assert abs(release_frame(p, "L") - 40) <= 1


def test_screen_ok():
    assert screen(synth(), "R") == (True, "")


def test_screen_camera_cut():
    p = synth()
    p.xy[60:, :, 0] += 0.6 * W   # whole person jumps >half the frame
    assert screen(p, "R") == (False, "camera_cut")


def test_screen_short():
    assert screen(synth(n=20), "R") == (False, "short")


def test_screen_no_pitcher_when_wrist_never_rises():
    p = synth()
    p.xy[:, 10, 1] = p.xy[:, 10, 1].max()   # wrist stays low: tracker is on a catcher/batter
    assert screen(p, "R") == (False, "no_pitcher")


def test_release_frame_first_peak_beats_larger_later_peak():
    """SYNTHETIC: release peak at 40, then a LARGER fielding/return-throw peak at 150."""
    p = synth(n=200)
    later = 60 * np.exp(-0.5 * ((np.arange(200) - 150) / 5.0) ** 2)
    p.xy[:, 10, 0] += np.cumsum(later)
    assert abs(release_frame(p, "R") - 40) <= 1
