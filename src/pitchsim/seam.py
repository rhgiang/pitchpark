"""Seam-shifted wake (Milestone 3): learn the push Magnus can't explain.

The seams trip the boundary layer on one side of the ball, so the wake shifts and the ball
gets an extra sideways/vertical force that Magnus + drag never model (Barton Smith's
experiments; Driveline's seam-shifted-wake write-ups). M1's calibration absorbs the average
of it per pitch type, but it still misses some types by inches in a consistent direction.

Target: for each pitch, the constant extra acceleration (x, z, ft/s^2) that makes the
calibrated M1 flight land on the Statcast fit crossing. It is "what's left after M1".
Caveat: release_state extrapolates Statcast's fit back to release and is off by ~0.1-0.25
ft/s, so the target also holds that start-state artefact; a learned push is not pure seam force.

Inputs are release-only (spin axis, spin rate, speed, release point, extension). Predicting
movement from movement (ax, az, pfx, plate_*) would be circular, so seam_design never reads
them; FORBIDDEN lists them and a test pins that.

Model: ridge regression (standardised features, so one penalty is fair), with the penalty
chosen by cross-validation grouped by game so pitches from one game never sit in both train
and validation folds. Spin axis enters as sin/cos so 359 and 1 degrees are neighbours.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pitchsim.statcast import fit_plate_crossing
from pitchsim.validation import run_sim

SEAM_FEATURES = ["sin_axis", "cos_axis", "release_spin_rate", "release_speed",
                 "release_pos_x", "release_pos_z", "release_extension"]
FORBIDDEN = ["ax", "ay", "az", "vx0", "vy0", "vz0", "pfx_x", "pfx_z", "plate_x", "plate_z"]
_PLAIN = SEAM_FEATURES[2:]


def seam_design(df: pd.DataFrame) -> np.ndarray:
    """N x 7 feature matrix in SEAM_FEATURES order, from release-only columns."""
    axis = np.radians(df["spin_axis"].to_numpy(float))
    return np.column_stack([np.sin(axis), np.cos(axis), df[_PLAIN].to_numpy(float)])


def push_target(df: pd.DataFrame, cd, eff) -> np.ndarray:
    """N x 2 constant push (ft/s^2) that closes each pitch's calibrated miss.

    Constant acceleration a over flight time t moves the ball a*t^2/2, so the first guess is
    a = -2*miss/t^2. Drag acts on the changed path, so that is only approximate: one
    correction pass re-simulates with a and adds the same formula for the remaining miss.
    cd and eff may be scalars or per-row arrays; df needs a 'rho' column.
    """
    fit_pos, _, _ = fit_plate_crossing(df)
    a = np.zeros((len(df), 2))
    for _ in range(2):
        pos, _, t = run_sim(df, cd, eff, "release", extra_acc=a)
        miss = pos[:, [0, 2]] - fit_pos[:, [0, 2]]
        a = a - 2 * miss / t[:, None] ** 2
    return a


@dataclass
class SeamModel:
    kind: str                      # "fixed", "per_pitch" or "fixed_fallback"
    mean: np.ndarray               # (2,) column means of the target
    pipeline: Pipeline | None
    alpha: float | None
    n_train: int
    n_games: int


def fit_fixed(target: np.ndarray) -> SeamModel:
    """Same push for every pitch of the type: the mean of the target."""
    return SeamModel("fixed", target.mean(axis=0), None, None, len(target), 0)


def fit_per_pitch(train: pd.DataFrame, target: np.ndarray, seed: int = 0) -> SeamModel:
    """Ridge on seam_design; alpha by game-grouped CV. One game cannot be grouped, so it
    falls back to the fixed mean (kind="fixed_fallback"). seed is unused: GroupKFold and
    Ridge are deterministic; kept so callers pass one like the other fits."""
    games = train["game_pk"].to_numpy()
    n_games = len(np.unique(games))
    if n_games < 2:
        m = fit_fixed(target)
        m.kind, m.n_games = "fixed_fallback", n_games
        return m
    search = GridSearchCV(
        Pipeline([("scale", StandardScaler()), ("ridge", Ridge())]),
        {"ridge__alpha": np.logspace(-3, 3, 13)},
        cv=GroupKFold(n_splits=min(5, n_games)), scoring="neg_mean_squared_error")
    search.fit(seam_design(train), target, groups=games)
    return SeamModel("per_pitch", target.mean(axis=0), search.best_estimator_,
                     float(search.best_params_["ridge__alpha"]), len(target), n_games)


def predict_push(model: SeamModel, df: pd.DataFrame) -> np.ndarray:
    """N x 2 predicted push (ft/s^2) for each row of df."""
    if model.pipeline is None:
        return np.tile(model.mean, (len(df), 1))
    return model.pipeline.predict(seam_design(df))
