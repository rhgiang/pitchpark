"""
2D comparison of the replica delivery with a real broadcast clip (M5 Task 8).

Idea: we have no 3D ground truth for a clip, but we do have its 2D keypoints (RTMPose, COCO-17).  So we
project the replica's 3D joints into that clip's image with the best camera we can fit, and measure how far
the projected joints land from the detected ones.  If the replica moves like Ohtani, the distance should be
about as small as when we put one of his OTHER real deliveries through the same procedure (the noise floor).

Camera model: an affine 3D->2D map (a 2x3 matrix A) plus a per-frame 2D translation b_f.
- Affine, not a pinhole: the centre-field camera sits ~400 ft away with a ~650 mm lens, so perspective
  across a 6 ft body is ~1% (body depth / distance): a weak-perspective (scaled orthographic) camera is the
  standard approximation (Hartley & Zisserman, Multiple View Geometry, sec. 6.3.4 "affine cameras").  A full
  2x3 affine also absorbs scale, aspect, and a small skew, so it never has to know the lens or distance.
- One A per clip (the lens and viewing direction barely change during a pitch), but one b per frame: the
  broadcast pans and the pitcher strides, and in a single 2D view those two cannot be told apart.  Note this
  also means the score is blind to the body's overall translation (stride length): it scores body SHAPE.
- Least squares: for a fixed A, the best b_f is the mean residual of frame f, so subtracting each frame's
  mean (over its confident joints) from both sides removes b exactly, and A is one linear solve.

Error unit: 2D distance / the clip's torso length (shoulder-mid to hip-mid, median over frames), a PCK-style
normalisation (Andriluka et al. 2014, "2D Human Pose Estimation: New Benchmark", PCKh uses head size the
same way), so clips filmed at different zooms are comparable.
"""
import numpy as np

# COCO-17 index -> SMPL-24 index, for the 12 joints both skeletons have (face skipped: SMPL has no eyes/ears,
# and its "head" joint is the skull centre, not the nose).  COCO 5 l_shoulder 6 r_shoulder 7 l_elbow 8 r_elbow
# 9 l_wrist 10 r_wrist 11 l_hip 12 r_hip 13 l_knee 14 r_knee 15 l_ankle 16 r_ankle.
COCO_TO_SMPL = {5: 16, 6: 17, 7: 18, 8: 19, 9: 20, 10: 21, 11: 1, 12: 2, 13: 4, 14: 5, 15: 7, 16: 8}
COCO_IDX = np.array(list(COCO_TO_SMPL))
SMPL_IDX = np.array(list(COCO_TO_SMPL.values()))
# positions in the 12-joint arrays
GROUPS = {"trunk": [0, 1, 6, 7], "arms": [2, 3, 4, 5], "legs": [8, 9, 10, 11]}
JOINT_NAMES = ["l_shoulder", "r_shoulder", "l_elbow", "r_elbow", "l_wrist", "r_wrist",
               "l_hip", "r_hip", "l_knee", "r_knee", "l_ankle", "r_ankle"]
PHASES = ["set_rocker", "leg_lift", "stride", "arm_action", "follow_through"]

# Why 6: a frame's translation is fit from its own joints, so a frame with only 2-3 confident joints would
# fit them almost perfectly by itself and read as a tiny error.  Half the 12 keeps the per-frame fit honest.
MIN_JOINTS_PER_FRAME = 6
# Why 0.5: RTMPose's SimCC scores for Ohtani's clips sit at 0.57-0.87 (5th-90th percentile); the ones under
# ~0.5 are mostly blurred wrists near release and occluded far-side joints, where the "detection" is a guess.
# Cost: the blurriest frames of the throwing arm count less, so the arm-action score is a bit flattering.
CONF_MIN = 0.5
# Winter (2009), Biomechanics and Motor Control of Human Movement, segment fractions of height H:
# shoulder (acromion) 0.818 H, hip (greater trochanter) 0.530 H -> shoulder-to-hip 0.288 H.
TORSO_FRAC_OF_HEIGHT = 0.288


def project(A: np.ndarray, b: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Image points (F, K, 2) px of 3D points X (F, K, 3) through camera A (2, 3) and per-frame pan b (F, 2)."""
    return X @ A.T + b[:, None, :]


def fit_camera(X: np.ndarray, x: np.ndarray, w: np.ndarray):
    """Least-squares affine camera A (2, 3) + per-frame translation b (F, 2) mapping X (F, K, 3) onto x (F, K, 2),
    using only points with w (F, K) True.  Frames with < MIN_JOINTS_PER_FRAME usable points: b NaN, ok False.
    Returns (A, b, ok)."""
    w = w & np.isfinite(X).all(-1) & np.isfinite(x).all(-1)
    ok = w.sum(1) >= MIN_JOINTS_PER_FRAME
    wm = w & ok[:, None]
    n = np.maximum(wm.sum(1, keepdims=True), 1)
    Xz, xz = np.where(wm[..., None], X, 0.0), np.where(wm[..., None], x, 0.0)
    Xm, xm = Xz.sum(1) / n, xz.sum(1) / n            # per-frame means of the used points
    P = (Xz - Xm[:, None])[wm]                       # centred 3D points (N, 3)
    Q = (xz - xm[:, None])[wm]                       # centred 2D points (N, 2)
    At, *_ = np.linalg.lstsq(P, Q, rcond=None)       # min-norm: a 2D reference padded with z = 0 still solves
    A = At.T
    b = xm - Xm @ A.T
    b[~ok] = np.nan
    return A, b, ok


def torso_px(x: np.ndarray, w: np.ndarray) -> float:
    """Median shoulder-mid to hip-mid length (px) over frames where all four are confident; x, w in the
    12-joint order (positions 0, 1 shoulders; 6, 7 hips)."""
    m = w[:, [0, 1, 6, 7]].all(1)
    d = np.linalg.norm((x[m, 0] + x[m, 1]) / 2 - (x[m, 6] + x[m, 7]) / 2, axis=-1)
    return float(np.median(d))


def score_sequence(X: np.ndarray, x: np.ndarray, w: np.ndarray, torso: float):
    """Fit the camera, then the per-joint error (F, K) in torso lengths (NaN where not confident / not fit).
    Returns (err, (A, b, ok))."""
    A, b, ok = fit_camera(X, x, w)
    err = np.linalg.norm(project(A, np.nan_to_num(b), X) - x, axis=-1) / torso
    err[~(w & ok[:, None]) | ~np.isfinite(err)] = np.nan
    return err, (A, b, ok)


def sample_frames(seq: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Linear interpolation of seq (T, ...) at fractional frame indices idx; NaN outside [0, T-1].
    Why linear (the viewer uses Catmull-Rom): at 60 fps the two differ by well under a pixel here."""
    idx = np.asarray(idx, float)
    inside = (idx >= 0) & (idx <= len(seq) - 1)
    i0 = np.clip(np.floor(idx).astype(int), 0, len(seq) - 1)
    i1 = np.minimum(i0 + 1, len(seq) - 1)
    f = (np.clip(idx, 0, len(seq) - 1) - i0).reshape((-1,) + (1,) * (seq.ndim - 1))
    out = (1 - f) * seq[i0] + f * seq[i1]
    out[~inside] = np.nan
    return out


def phase_index(t_s: np.ndarray, bounds_s) -> np.ndarray:
    """Phase number of each time (s from release): phase k is (bounds[k-1], bounds[k]], so a frame exactly
    on a boundary closes the earlier phase (release, t = 0, is the last frame of the arm action)."""
    return np.searchsorted(np.asarray(bounds_s), np.asarray(t_s), side="left")


def boot_median_ci(errs_by_clip, games, n: int = 1000, seed: int = 0):
    """Median of all pooled errors, with a 95% interval from resampling GAMES (clips of one game share
    a camera set-up and a day, so games are the independent units).  Returns (median, lo, hi)."""
    rng = np.random.default_rng(seed)
    games = np.asarray(games)
    by = {g: np.concatenate([e[np.isfinite(e)] for e, gg in zip(errs_by_clip, games) if gg == g])
          for g in np.unique(games)}
    keys = list(by)
    stats = [np.median(np.concatenate([by[keys[i]] for i in rng.integers(0, len(keys), len(keys))]))
             for _ in range(n)]
    allv = np.concatenate(list(by.values()))
    return float(np.median(allv)), *map(float, np.quantile(stats, [0.025, 0.975]))


# A camera cut changes the pitcher's image size abruptly (a batter close-up, a wide shot).  Why 1.5x: within
# one shot the torso's 2D length changes slowly with lean (well under 1.5x over the windup); a cut to a
# close-up is ~3x.  Why 3 frames: a single mis-detected frame is not a cut.
CUT_RATIO = 1.5
CUT_RUN = 3


def shot_start(x: np.ndarray, w: np.ndarray, release_frame: int, fps: float) -> int:
    """First frame of the camera shot that contains the release: walk back from release until the torso's
    2D length is off its stride-phase median (1.0-0.3 s before release) by > CUT_RATIO for CUT_RUN frames
    in a row.  Frames before it show another shot (often the batter), where neither the 2D track nor the
    lift is the pitcher.  x, w in the 12-joint order."""
    m = w[:, [0, 1, 6, 7]].all(1)
    tor = np.linalg.norm((x[:, 0] + x[:, 1]) / 2 - (x[:, 6] + x[:, 7]) / 2, axis=-1)
    tor[~m] = np.nan
    ref = np.nanmedian(tor[max(release_frame - int(1.0 * fps), 0): max(release_frame - int(0.3 * fps), 1)])
    run, first = 0, None
    for f in range(release_frame, -1, -1):
        r = tor[f] / ref
        if not np.isfinite(r):
            continue  # unconfident frame: no evidence either way (a close-up often has no confident hips)
        if 1 / CUT_RATIO <= r <= CUT_RATIO:
            run, first = 0, None
            continue
        run, first = run + 1, first if run else f
        if run == CUT_RUN:
            return first + 1
    return 0


def image_arm_angle_deg(shoulder_xy: np.ndarray, point_xy: np.ndarray, side: int) -> np.ndarray:
    """Image-plane angle (deg) of shoulder -> elbow/wrist, 0 = horizontal toward the throwing side, +90 = up.

    x, y in px (y DOWN).  side = +1 if the throwing side is image-right, -1 if image-left (the x axis is
    mirrored so every pitcher's arm reads in the same convention, the catcher's-view arm angle of Statcast's
    `arm_angle`).  Range (-180, 180]."""
    d = np.asarray(point_xy, float) - np.asarray(shoulder_xy, float)
    return np.degrees(np.arctan2(-d[..., 1], side * d[..., 0]))
