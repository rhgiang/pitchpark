"""
Per-pitch-type generative model of how a pitch leaves the hand (M2).

One Gaussian (or small Gaussian mixture) per pitch type over release position,
extension, speed, spin rate, spin axis and the two release angles.
"""

import copy
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from pitchsim.statcast import release_state

# Features scored against real pitches.
GRADED = ["release_pos_x", "release_pos_z", "release_extension",
          "release_speed", "release_spin_rate", "spin_axis"]
# Release angles are modelled but not graded: they are where he aims, which is M6's job.
# They are still needed so the sampled velocity direction is consistent with the release point.
ANGLES = ["release_angle_v_deg", "release_angle_h_deg"]
FEATURES = GRADED + ANGLES

# Too few pitches and an 8-D covariance (36 parameters) is mostly noise.
MIN_TRAIN = 30
MIN_TEST = 30
# A mixture needs enough data per component to be worth its extra parameters.
MIN_MIXTURE = 100
MAX_COMPONENTS = 4
# Number of generated sets averaged in the classifier test, and the AUC counted as a pass.
N_DRAWS = 10
PASS_AUC = 0.55
N_BOOT = 200


def add_release_angles(df: pd.DataFrame) -> pd.DataFrame:
    """Copy of df with vertical / horizontal release angles (degrees) from release velocity.

    y points toward the pitcher, so the ball travels along -vy: the vertical angle is
    atan2(vz, -vy) (negative = thrown downward), the horizontal one atan2(vx, -vy).
    """
    _, vel = release_state(df)
    out = df.copy()
    out["release_angle_v_deg"] = np.degrees(np.arctan2(vel[:, 2], -vel[:, 1]))
    out["release_angle_h_deg"] = np.degrees(np.arctan2(vel[:, 0], -vel[:, 1]))
    return out


def circular_mean_deg(a: np.ndarray) -> float:
    """Mean direction of angles in degrees, in [0, 360)."""
    r = np.radians(np.asarray(a, dtype=float))
    return float(np.degrees(np.arctan2(np.sin(r).mean(), np.cos(r).mean())) % 360)


def unwrap_deg(a: np.ndarray, center: float) -> np.ndarray:
    """Shift angles (degrees) by multiples of 360 into center +/- 180.

    Spin axis is circular, so 359 and 1 are 2 deg apart. A plain mean of them is 180
    and the variance is huge; unwrapping around a circular mean makes a Gaussian fit.
    """
    return (np.asarray(a, dtype=float) - center + 180) % 360 - 180 + center


@dataclass
class PitchTypeModel:
    gmm: GaussianMixture
    axis_center_deg: float  # spin_axis was unwrapped around this before fitting
    n_train: int


def fit_pitch_type(train: pd.DataFrame, mixture: bool, seed: int = 0) -> PitchTypeModel:
    """Fit a Gaussian (mixture=False) or BIC-chosen mixture to one pitch type's FEATURES."""
    X = train[FEATURES].to_numpy(dtype=float).copy()
    center = circular_mean_deg(X[:, FEATURES.index("spin_axis")])
    X[:, FEATURES.index("spin_axis")] = unwrap_deg(X[:, FEATURES.index("spin_axis")], center)

    # Full covariance: release height and extension (and speed / spin) move together,
    # so a diagonal model would sample combinations he never throws.
    def fit(k: int) -> GaussianMixture:
        return GaussianMixture(k, covariance_type="full", random_state=seed).fit(X)

    if mixture and len(X) >= MIN_MIXTURE:
        # BIC penalises each extra component's parameters, so it stops adding
        # components once they only fit noise (log-likelihood alone always improves).
        gmm = min((fit(k) for k in range(1, MAX_COMPONENTS + 1)), key=lambda g: g.bic(X))
    else:
        gmm = fit(1)
    return PitchTypeModel(gmm, center, len(X))


def sample(model: PitchTypeModel, n: int, seed: int) -> pd.DataFrame:
    """Draw n synthetic pitches (columns FEATURES); spin_axis wrapped back to [0, 360)."""
    g = copy.copy(model.gmm)  # sklearn draws with its own random_state; don't mutate the model
    g.random_state = seed
    out = pd.DataFrame(g.sample(n)[0], columns=FEATURES)
    out["spin_axis"] = out["spin_axis"] % 360
    return out


def _design(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    """Feature matrix with spin_axis replaced by sin and cos of the angle.

    Raw axis jumps from 360 to 0, so a classifier could "detect" fakes by that
    artificial gap; sin/cos puts 359 and 1 next to each other.
    """
    parts = []
    for c in cols:
        if c == "spin_axis":
            r = np.radians(df[c].to_numpy(dtype=float))
            parts += [np.sin(r), np.cos(r)]
        else:
            parts.append(df[c].to_numpy(dtype=float))
    return np.column_stack(parts)


def classifier_auc(real: pd.DataFrame, generated: list[pd.DataFrame], cols: list[str],
                   center: float, seed: int = 0) -> tuple[float, float, float]:
    """Classifier two-sample test: can a model tell real (0) from generated (1)?

    Returns (mean AUC, mean 2.5% bound, mean 97.5% bound) over the generated sets;
    AUC near 0.5 means indistinguishable. Each set should be the same size as `real`.
    `center` is unused here (sin/cos needs no unwrapping); it is kept so the signature
    matches marginal_ks.

    Why a nonlinear classifier: a fitted Gaussian matches means and linear correlations
    by construction, so logistic regression would pass it trivially. Gradient boosting
    can find what is left (curved or clumped structure). Out-of-fold predictions keep
    the classifier from scoring itself on pitches it trained on.
    Why average several sets: one draw of a few dozen pitches gives a noisy AUC.
    Ref: Lopez-Paz & Oquab, "Revisiting Classifier Two-Sample Tests" (ICLR 2017).
    """
    rng = np.random.default_rng(seed)
    Xr = _design(real, cols)
    aucs, los, his = [], [], []
    for gen in generated:
        X = np.vstack([Xr, _design(gen, cols)])
        y = np.r_[np.zeros(len(Xr)), np.ones(len(gen))].astype(int)
        p = cross_val_predict(HistGradientBoostingClassifier(random_state=seed), X, y,
                              cv=StratifiedKFold(5, shuffle=True, random_state=seed),
                              method="predict_proba")[:, 1]
        aucs.append(roc_auc_score(y, p))
        # Bootstrap the predictions to see how much the AUC could move by chance.
        boot = []
        for _ in range(N_BOOT):
            i = rng.integers(0, len(y), len(y))
            if len(np.unique(y[i])) == 2:
                boot.append(roc_auc_score(y[i], p[i]))
        los.append(np.percentile(boot, 2.5)); his.append(np.percentile(boot, 97.5))
    return float(np.mean(aucs)), float(np.mean(los)), float(np.mean(his))


def marginal_ks(real: pd.DataFrame, generated: pd.DataFrame, center: float) -> pd.DataFrame:
    """Per-feature two-sample KS test of GRADED columns; flagged when p < 0.01.

    Spin axis is unwrapped around `center` first so a cluster straddling 0/360
    is not read as two distant groups.
    """
    rows = []
    for c in GRADED:
        a, b = real[c].to_numpy(dtype=float), generated[c].to_numpy(dtype=float)
        if c == "spin_axis":
            a, b = unwrap_deg(a, center), unwrap_deg(b, center)
        r = ks_2samp(a, b)
        rows.append({"feature": c, "ks_stat": r.statistic, "p_value": r.pvalue,
                     "flagged": bool(r.pvalue < 0.01)})
    return pd.DataFrame(rows)
