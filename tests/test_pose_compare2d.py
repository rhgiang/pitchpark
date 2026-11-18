"""Task 8 tests, compare2d.py.  ALL INPUTS ARE SYNTHETIC (seeded random 3D "joints", a known affine
camera, a random pan, pixel noise), not real pitcher data."""
import numpy as np

from pitchsim.pose.compare2d import (COCO_IDX, SMPL_IDX, boot_median_ci, fit_camera, phase_index, project,
                                     sample_frames, score_sequence, torso_px)


def _synthetic(rng, F=60, K=12):
    """A moving synthetic 3D body (ft) and its 2D image (px) through a known affine camera + per-frame pan."""
    base = rng.normal(0, 2.0, (K, 3))
    t = np.linspace(0, 1, F)[:, None, None]
    X = base[None] + np.sin(2 * np.pi * t * rng.uniform(0.5, 2, (1, K, 3))) * 0.8
    A = np.array([[60.0, 4.0, -3.0], [2.0, -5.0, -58.0]])     # ~60 px/ft, image y down
    b = np.cumsum(rng.normal(0, 3, (F, 2)), axis=0) + [640, 360]  # broadcast pan: a random walk
    return X, A, b


def test_fit_recovers_known_camera():
    rng = np.random.default_rng(0)
    X, A, b = _synthetic(rng)
    x = project(A, b, X) + rng.normal(0, 0.5, (X.shape[0], X.shape[1], 2))
    w = np.ones(x.shape[:2], bool)
    A_hat, b_hat, ok = fit_camera(X, x, w)
    assert ok.all()
    np.testing.assert_allclose(A_hat, A, atol=0.2)
    np.testing.assert_allclose(b_hat, b, atol=1.0)


def test_masked_points_are_ignored():
    rng = np.random.default_rng(1)
    X, A, b = _synthetic(rng)
    x = project(A, b, X)
    w = rng.random(x.shape[:2]) > 0.2
    x[~w] += 500.0  # garbage where not confident
    A_hat, _, _ = fit_camera(X, x, w)
    np.testing.assert_allclose(A_hat, A, atol=1e-6)


def test_frames_with_too_few_joints_are_not_fit():
    rng = np.random.default_rng(2)
    X, A, b = _synthetic(rng)
    x = project(A, b, X)
    w = np.ones(x.shape[:2], bool)
    w[5, 3:] = False  # only 3 confident joints in frame 5
    _, b_hat, ok = fit_camera(X, x, w)
    assert not ok[5] and np.isnan(b_hat[5]).all() and ok.sum() == len(ok) - 1


def test_matched_pose_scores_near_zero_mismatched_worse():
    rng = np.random.default_rng(3)
    X, A, b = _synthetic(rng)
    x = project(A, b, X) + rng.normal(0, 0.5, (X.shape[0], X.shape[1], 2))
    w = np.ones(x.shape[:2], bool)
    e_good, _ = score_sequence(X, x, w, torso=100.0)
    X_bad = X[::-1].copy()  # same body, motion played backwards: a mismatched delivery
    e_bad, _ = score_sequence(X_bad, x, w, torso=100.0)
    X_static = np.repeat(X[:1], len(X), axis=0)  # the naive baseline: one pose held
    e_static, _ = score_sequence(X_static, x, w, torso=100.0)
    assert np.nanmedian(e_good) < 0.01
    assert np.nanmedian(e_bad) > 10 * np.nanmedian(e_good)
    assert np.nanmedian(e_static) > 10 * np.nanmedian(e_good)


def test_2d_reference_padded_with_zero_depth_fits():
    """The noise floor compares 2D to 2D by padding z = 0: the affine fit becomes a 2D affine."""
    rng = np.random.default_rng(4)
    X, A, b = _synthetic(rng)
    ref = project(np.array([[1.1, 0.1, 0.0], [0.0, 0.9, 0.0]]), np.zeros((len(X), 2)), X)
    x = ref @ np.array([[0.8, 0.2], [-0.1, 1.2]]).T + b[:, None]
    e, _ = score_sequence(np.concatenate([ref, np.zeros(ref.shape[:2] + (1,))], -1), x,
                          np.ones(x.shape[:2], bool), torso=100.0)
    assert np.nanmax(e) < 1e-6


def test_torso_px_and_joint_maps():
    x = np.zeros((3, 12, 2))
    x[:, 0], x[:, 1] = [0, 0], [10, 0]      # shoulders (COCO 5, 6)
    x[:, 6], x[:, 7] = [0, 40], [10, 40]    # hips (COCO 11, 12)
    assert torso_px(x, np.ones((3, 12), bool)) == 40.0
    assert len(COCO_IDX) == len(SMPL_IDX) == 12
    assert list(COCO_IDX[:2]) == [5, 6] and list(SMPL_IDX[:2]) == [16, 17]  # l/r shoulder both ways


def test_sample_frames_interpolates_and_blanks_outside():
    seq = np.arange(5, dtype=float)[:, None, None] * np.ones((5, 2, 3))
    out = sample_frames(seq, np.array([-1.0, 0.0, 1.5, 4.0, 4.5]))
    assert np.isnan(out[0]).all() and np.isnan(out[4]).all()
    np.testing.assert_allclose(out[1:4, 0, 0], [0.0, 1.5, 4.0])


def test_phase_index():
    bounds = [-1.0, -0.5, 0.0]  # 4 phases, each (previous bound, bound]: release (t = 0) ends arm action
    assert list(phase_index(np.array([-2.0, -1.0, -0.6, 0.0, 0.01]), bounds)) == [0, 0, 1, 2, 3]


def test_boot_ci_contains_median():
    rng = np.random.default_rng(5)
    errs = [rng.random(50) for _ in range(8)]
    med, lo, hi = boot_median_ci(errs, [0, 0, 1, 1, 2, 2, 3, 3])
    assert lo <= med <= hi


def test_shot_start_finds_a_camera_cut():
    """Synthetic: a body 3x bigger (another shot, a close-up) for frames 0-29, the pitcher shot after."""
    from pitchsim.pose.compare2d import shot_start
    F = 120
    x = np.zeros((F, 12, 2))
    x[:, 6:8, 1] = 40.0                       # hips 40 px below the shoulders
    x[:30, 6:8, 1] = 120.0                    # close-up before the cut
    w = np.ones((F, 12), bool)
    w[50, :] = False                          # a missing frame is not a cut
    x[70, 6:8, 1] = 90.0                      # nor is a single bad frame
    assert shot_start(x, w, release_frame=110, fps=60.0) == 30
    assert shot_start(x[30:], w[30:], release_frame=80, fps=60.0) == 0


def test_image_arm_angle_deg_synthetic():
    """Synthetic points: image y points DOWN, so a hand 1 px right and 1 px up of the shoulder is +45 deg; a
    hand on the other side (side = -1, a mirrored thrower) is read as if mirrored, so the same arm is still +45."""
    from pitchsim.pose.compare2d import image_arm_angle_deg
    sh = np.array([[100.0, 100.0], [100.0, 100.0]])
    hand = np.array([[101.0, 99.0], [99.0, 99.0]])
    assert np.allclose(image_arm_angle_deg(sh, hand, 1), [45.0, 135.0])
    assert np.allclose(image_arm_angle_deg(sh, hand, -1), [135.0, 45.0])  # mirrored
    assert np.allclose(image_arm_angle_deg(sh, sh + [0.0, 5.0], 1), -90.0)     # straight down
