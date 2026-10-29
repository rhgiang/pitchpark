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
MIN_PITCHES = 30
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


def run_sim(df, cd, eff, start):
    """Simulate each row to the plate; df must have a 'rho' column (lb/ft^3)."""
    pos0, vel0 = release_state(df) if start == "release" else initial_state(df)
    return simulate_batch(pos0, vel0, df["release_spin_rate"].to_numpy(),
                          df["spin_axis"].to_numpy(), cd, eff, rho=df["rho"].to_numpy())


def residuals(df, fit_vel, cd, eff, start):
    pos, vel, _ = run_sim(df, cd, eff, start)
    dx = (pos[:, 0] - df["plate_x"].to_numpy()) * 12
    dz = (pos[:, 2] - df["plate_z"].to_numpy()) * 12
    dv = (np.linalg.norm(vel, axis=1) - np.linalg.norm(fit_vel, axis=1)) * FT_PER_S_TO_MPH
    return dx, dz, dv


def calibrate(train, start):
    _, fit_vel, _ = fit_plate_crossing(train)

    def resid(params):
        dx, dz, dv = residuals(train, fit_vel, params[0], params[1], start)
        return np.concatenate([dx, dz, dv])   # inches and mph, roughly comparable

    res = least_squares(resid, x0=[0.33, 0.9], bounds=([0.15, 0.2], [0.6, 1.0]),
                        diff_step=1e-3)
    return res.x


def angle_diff(a, b):
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
        _, test_fit_vel, _ = fit_plate_crossing(test)

        d = PhysicsParams()
        bx, bz, bv = residuals(test, test_fit_vel, d.cd, d.spin_efficiency, start)
        cd, eff = calibrate(train, start)
        cx, cz, cv = residuals(test, test_fit_vel, cd, eff, start)

        rows.append({
            "pitch_type": ptype,
            "pitch_name": g["pitch_name"].iloc[0] if "pitch_name" in g else ptype,
            "n": len(g),
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
