"""
Helpers for turning Statcast rows into simulator inputs.

Statcast describes each pitch with a 9-parameter constant-acceleration fit,
defined at y = 50 ft:
    vx0, vy0, vz0   velocity at y = 50 ft (ft/s)
    ax, ay, az      constant acceleration over the whole flight (ft/s^2)
plus release_pos_x/y/z, plate_x/plate_z (at the front of home plate),
release_spin_rate (rpm) and spin_axis (deg, inferred from movement).

The x/z position at y = 50 ft is not in the CSV, so we recover it by running
the fit backwards from the release point.
"""

import numpy as np
import pandas as pd

from pitchsim.physics import FRONT_OF_PLATE_Y, G, STATCAST_Y0

REQUIRED = ["pitch_type", "release_speed", "release_pos_x", "release_pos_y",
            "release_pos_z", "vx0", "vy0", "vz0", "ax", "ay", "az",
            "plate_x", "plate_z", "release_spin_rate", "spin_axis"]


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Keep pitches with complete tracking data and a known pitch type."""
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing Statcast columns: {missing}")
    out = df.dropna(subset=REQUIRED)
    out = out[~out["pitch_type"].isin(["PO", "IN", "UN", "AB"])]  # pickoffs, intentional, unknown
    return out.reset_index(drop=True)


def _time_to_y(y_target, vy0, ay):
    """Time (from the y = 50 ft reference) at which the fit reaches y_target.

    Solves 0.5*ay*t^2 + vy0*t + (y0 - y_target) = 0 and picks the physical root.
    Works for both directions: negative t for the release point (y > 50),
    positive t for the plate (y < 50).
    """
    c = STATCAST_Y0 - y_target
    disc = vy0 ** 2 - 2 * ay * c
    return (-vy0 - np.sqrt(disc)) / ay


def initial_state(df: pd.DataFrame):
    """Position and velocity at y = 50 ft for each pitch (N x 3 arrays)."""
    vx0, vy0, vz0 = df["vx0"].to_numpy(), df["vy0"].to_numpy(), df["vz0"].to_numpy()
    ax, ay, az = df["ax"].to_numpy(), df["ay"].to_numpy(), df["az"].to_numpy()
    t_rel = _time_to_y(df["release_pos_y"].to_numpy(), vy0, ay)   # negative
    x0 = df["release_pos_x"].to_numpy() - vx0 * t_rel - 0.5 * ax * t_rel ** 2
    z0 = df["release_pos_z"].to_numpy() - vz0 * t_rel - 0.5 * az * t_rel ** 2
    pos0 = np.stack([x0, np.full_like(x0, STATCAST_Y0), z0], axis=1)
    vel0 = np.stack([vx0, vy0, vz0], axis=1)
    return pos0, vel0


def release_state(df: pd.DataFrame):
    """Position and velocity at the release point for each pitch (N x 3 arrays).

    Statcast's constant-acceleration fit is anchored at y = 50 ft but is valid
    over the whole flight, so evaluating it at release_pos_y (t_rel < 0) gives
    the release velocity: v = v0 + a * t_rel. The position is release_pos_*
    by construction (initial_state derived its y = 50 position from it), so
    matching it is not independent evidence.
    """
    t_rel = _time_to_y(df["release_pos_y"].to_numpy(), df["vy0"].to_numpy(), df["ay"].to_numpy())
    vel = np.stack([df[v].to_numpy() + df[a].to_numpy() * t_rel
                    for v, a in (("vx0", "ax"), ("vy0", "ay"), ("vz0", "az"))], axis=1)
    pos = df[["release_pos_x", "release_pos_y", "release_pos_z"]].to_numpy()
    return pos, vel


def fit_plate_crossing(df: pd.DataFrame):
    """Where Statcast's own constant-acceleration fit crosses the plate.

    Should reproduce plate_x / plate_z almost exactly; it's a check that we
    are reading the data and coordinates correctly.
    """
    pos0, vel0 = initial_state(df)
    acc = df[["ax", "ay", "az"]].to_numpy()
    t = _time_to_y(FRONT_OF_PLATE_Y, vel0[:, 1], acc[:, 1])
    pos = pos0 + vel0 * t[:, None] + 0.5 * acc * t[:, None] ** 2
    vel = vel0 + acc * t[:, None]
    return pos, vel, t


def fit_trajectory(row: pd.Series, n: int = 100) -> np.ndarray:
    """Points along Statcast's fitted path from release to plate (for plots)."""
    one = row.to_frame().T.astype({c: float for c in REQUIRED if c != "pitch_type"})
    pos0, vel0 = initial_state(one)
    acc = one[["ax", "ay", "az"]].to_numpy()[0]
    t0 = _time_to_y(float(row["release_pos_y"]), vel0[0, 1], acc[1])
    t1 = _time_to_y(FRONT_OF_PLATE_Y, vel0[0, 1], acc[1])
    ts = np.linspace(t0, t1, n)[:, None]
    return pos0[0] + vel0[0] * ts + 0.5 * acc * ts ** 2


def observed_break_direction_deg(df: pd.DataFrame) -> np.ndarray:
    """Direction of the non-gravity acceleration in the x-z plane, as an angle
    in Statcast's spin_axis convention. Used to sanity-check the sign
    convention in physics.spin_unit_vector against real data."""
    mx = df["ax"].to_numpy()
    mz = df["az"].to_numpy() + G
    # invert m = (sin th, -cos th)
    return np.degrees(np.arctan2(mx, -mz)) % 360
