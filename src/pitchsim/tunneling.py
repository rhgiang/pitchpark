"""
Look / deception metrics (M4): approach angles, perceived velocity, tunnel separation.

Coordinates as in Statcast (feet; x toward catcher's right, y toward the pitcher, z up).
"""
from __future__ import annotations

import numpy as np

# Pitcher's rubber is 60.5 ft from the plate's back tip; Statcast's release_extension is
# measured from the rubber toward the plate, so flight distance = 60.5 - extension.
RUBBER_Y_FT = 60.5

# Hitter's decision point: ~23.6 ft in front of the plate's front edge (y = 25 ft in the
# plate-back-tip frame, i.e. inside the 23-25 ft range). The ball is ~0.167 s from the plate;
# after that a hitter cannot meaningfully change the swing (Baseball Prospectus tunneling
# articles, "tunnel point", ~20-25 ft).
DECISION_Y_FT = 25.0

# Reference extension for perceived velocity. Statcast describes effective speed as the
# speed "perceived by the batter" given extension, relative to an average pitcher. We use
# the same time-equivalent definition. Statcast's reference isn't published; Task 2 assumed
# 6.0 ft, which ran ~0.6 mph above `effective_speed` for all five pitchers. 6.34 ft is the value
# that best fits `effective_speed` on the TRAINING games of the five M3 pitchers pooled (grid
# search, median |diff|; scripts/m4_checks.py prints it). On held-out games it cuts the median
# |diff| from 0.63 to 0.25 mph (results/M4.md). Fitted on nothing but that comparison.
PERCEIVED_REF_EXT_FT = 6.34


def approach_angles_deg(vel_plate: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(VAA, HAA) in degrees for N x 3 plate velocities (ft/s).

    The approach angle is the direction of motion the hitter's eyes must track at the plate.
    VAA is negative for a descending ball (Savant's sign); HAA is positive when moving
    toward +x. The ball travels in -y, hence the -vy.
    """
    v = np.atleast_2d(np.asarray(vel_plate, dtype=float))
    return (np.degrees(np.arctan2(v[:, 2], -v[:, 1])),
            np.degrees(np.arctan2(v[:, 0], -v[:, 1])))


def perceived_velocity_mph(speed_mph, extension_ft) -> np.ndarray:
    """Time-equivalent speed: the speed that covers the reference flight distance
    (60.5 - PERCEIVED_REF_EXT_FT) in the time this pitch covers (60.5 - extension_ft).

    Longer extension shortens the flight, leaving the hitter less time, so it "plays" faster.
    """
    return (np.asarray(speed_mph, dtype=float)
            * (RUBBER_Y_FT - PERCEIVED_REF_EXT_FT) / (RUBBER_Y_FT - np.asarray(extension_ft, dtype=float)))


def position_at_y(t: np.ndarray, pos: np.ndarray, y_ft: float) -> np.ndarray:
    """(x, z) where a sampled path (times t, N x 3 positions) crosses y = y_ft.

    Linear between samples; y must decrease along the path (ball moves toward the plate).
    """
    pos = np.asarray(pos, dtype=float)
    y = pos[:, 1]
    if not (y[-1] <= y_ft <= y[0]):
        raise ValueError(f"y={y_ft} outside path range [{y[-1]}, {y[0]}]")
    # np.interp needs increasing x, so flip.
    return np.array([np.interp(y_ft, y[::-1], pos[::-1, k]) for k in (0, 2)])


def separation_in(path_a: tuple[np.ndarray, np.ndarray], path_b: tuple[np.ndarray, np.ndarray],
                  y_ft: float = DECISION_Y_FT) -> float:
    """Distance (inches) between two paths' (x, z) at y_ft. Each path is (t, N x 3 pos)."""
    return float(np.linalg.norm(position_at_y(*path_a, y_ft) - position_at_y(*path_b, y_ft)) * 12.0)
