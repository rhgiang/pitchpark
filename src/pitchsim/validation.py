"""
Calibrate Cd / spin efficiency per pitch type and score them on held-out games.
"""

from typing import Literal

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from pitchsim.physics import PhysicsParams, simulate_batch
from pitchsim.statcast import fit_plate_crossing, initial_state, release_state

FT_PER_S_TO_MPH = 0.681818
# Fewer pitches than this and a 2-parameter fit (plus a 30% held-out slice) is mostly noise.
MIN_PITCHES = 30
# Speed cap: every optimizer step re-simulates all training pitches, and Cd / efficiency
# stop moving well before 400 pitches.
MAX_TRAIN = 400


def split_by_game(df: pd.DataFrame, train_frac: float = 0.7,
                  seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split whole games into train/test.

    Pitches from one game share air density, mound and umpire, so a random
    pitch-level split leaks that shared context into the test set and flatters
    the held-out error. Splitting by game keeps the test set truly unseen.
    """
    games = np.random.default_rng(seed).permutation(df["game_pk"].unique())
    train_games = games[:int(np.ceil(train_frac * len(games)))]
    in_train = df["game_pk"].isin(train_games)
    return df[in_train], df[~in_train]


def run_sim(df: pd.DataFrame, cd: float, eff: float,
            start: Literal["y50", "release"], extra_acc: np.ndarray | None = None
            ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate each row to the front of the plate from the y = 50 ft or release state.

    df must have a 'rho' column (lb/ft^3). extra_acc is an optional (N, 2) constant push
    in ft/s^2. Returns (pos, vel, flight time) like simulate_batch.
    """
    pos0, vel0 = release_state(df) if start == "release" else initial_state(df)
    return simulate_batch(pos0, vel0, df["release_spin_rate"].to_numpy(),
                          df["spin_axis"].to_numpy(), cd, eff, rho=df["rho"].to_numpy(),
                          extra_acc=extra_acc)


def residuals(df: pd.DataFrame, fit_pos: np.ndarray, fit_vel: np.ndarray, cd: float,
              eff: float, start: Literal["y50", "release"],
              extra_acc: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulated minus Statcast-fit plate crossing: dx, dz (inches) and speed (mph).

    The target is the fit's own crossing at FRONT_OF_PLATE_Y (from fit_plate_crossing),
    not plate_x / plate_z. Statcast has changed where it reports plate_x / plate_z:
    in our 2026 downloads it matches the fit at y ~ 0.71 ft, not 17/12 ft, and scoring a
    17/12 simulation against it adds a built-in ~1 in error. The fit crossing is the same
    in every season. validate.py's check [0] still compares the two, as an alarm.
    """
    pos, vel, _ = run_sim(df, cd, eff, start, extra_acc)
    dx = (pos[:, 0] - fit_pos[:, 0]) * 12
    dz = (pos[:, 2] - fit_pos[:, 2]) * 12
    dv = (np.linalg.norm(vel, axis=1) - np.linalg.norm(fit_vel, axis=1)) * FT_PER_S_TO_MPH
    return dx, dz, dv


def calibrate(train: pd.DataFrame, start: Literal["y50", "release"],
              eff_min: float = 0.01) -> np.ndarray:
    """Least-squares fit of [Cd, spin efficiency] on training pitches.

    eff_min is efficiency's lower bound. Only scripts/seam_bound_check.py changes it, to compare
    with the old 0.20 bound.
    """
    fit_pos, fit_vel, _ = fit_plate_crossing(train)

    def resid(params):
        dx, dz, dv = residuals(train, fit_pos, fit_vel, params[0], params[1], start)
        # 1 in of miss weighs the same as 1 mph of speed error: both are about the size of
        # a typical residual, so neither term drowns the other.
        return np.concatenate([dx, dz, dv])

    # Bounds: Cd 0.15-0.6 brackets every published baseball drag value with room to spare.
    # Efficiency 0.01-1.0: 1.0 is all spin transverse, and a positive floor keeps a share of
    # spin a share. The floor was 0.20 until 2026-11-04, but that forced excess spin force on
    # some sliders and cutters; their fitted values land at about 0.07-0.15 on real data.
    res = least_squares(resid, x0=[0.33, 0.9], bounds=([0.15, eff_min], [0.6, 1.0]),
                        diff_step=1e-3)
    return res.x


def angle_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Signed difference a - b in degrees, wrapped to [-180, 180)."""
    return (a - b + 180) % 360 - 180


def evaluate(df: pd.DataFrame, start: Literal["y50", "release"],
             rho: np.ndarray) -> tuple[pd.DataFrame, list[str]]:
    """Per pitch type: baseline vs calibrated error on held-out games.

    rho is per pitch, aligned with df's rows. Returns (rows, skipped reasons).
    """
    df = df.assign(rho=rho).reset_index(drop=True)
    train_all, test_all = split_by_game(df)
    rows, skipped = [], []
    for ptype, g in df.groupby("pitch_type"):
        train = train_all[train_all["pitch_type"] == ptype]
        test = test_all[test_all["pitch_type"] == ptype]
        if len(g) < MIN_PITCHES:
            skipped.append(f"{ptype} (n={len(g)}): fewer than {MIN_PITCHES} pitches")
            continue
        if len(train) == 0 or len(test) == 0:
            skipped.append(f"{ptype} (n={len(g)}): no pitches in "
                           f"{'training' if len(train) == 0 else 'held-out'} games")
            continue
        if len(train) > MAX_TRAIN:
            train = train.sample(MAX_TRAIN, random_state=0)
        test_pos, test_vel, _ = fit_plate_crossing(test)

        d = PhysicsParams()
        bx, bz, bv = residuals(test, test_pos, test_vel, d.cd, d.spin_efficiency, start)
        cd, eff = calibrate(train, start)
        cx, cz, cv = residuals(test, test_pos, test_vel, cd, eff, start)

        rows.append({
            "pitch_type": ptype,
            "pitch_name": g["pitch_name"].iloc[0] if "pitch_name" in g else ptype,
            "n": len(g),
            "n_train": len(train),   # after the MAX_TRAIN cap
            "n_test": len(test),
            "mph": g["release_speed"].mean(),
            "rpm": g["release_spin_rate"].mean(),
            "baseline_err_in": np.median(np.hypot(bx, bz)),
            "cd": cd,
            "spin_eff": eff,
            "calibrated_err_in": np.median(np.hypot(cx, cz)),
            "p90_err_in": np.percentile(np.hypot(cx, cz), 90),
            "bias_x_in": cx.mean(),
            "bias_z_in": cz.mean(),
            "plate_speed_err_mph": np.median(np.abs(cv)),
        })
    return pd.DataFrame(rows), skipped
