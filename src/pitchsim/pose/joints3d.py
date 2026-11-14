"""
3D lift output: load/save only.  Pure numpy, no torch (the lifter runs in .venv-pose via
scripts/m5_lift.py and writes these files).

joints_m is (F, 24, 3) metres in SMPL joint order (identical to JOINTS in web/src/skeleton.js);
`up` is the gravity-up unit vector in the same frame; `lifter` names the model + settings used.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np

N_JOINTS = 24  # SMPL


@dataclass
class Lift3D:
    joints_m: np.ndarray  # (F, 24, 3) float32, metres
    fps: float
    up: np.ndarray        # (3,) unit vector, gravity-up in the joints_m frame
    lifter: str


def save_lift(lift: Lift3D, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, joints_m=lift.joints_m.astype(np.float32), fps=lift.fps,
                        up=np.asarray(lift.up, dtype=np.float64), lifter=lift.lifter)


def load_lift(path) -> Lift3D:
    z = np.load(path)
    j = z["joints_m"]
    if j.ndim != 3 or j.shape[1] != N_JOINTS or j.shape[2] != 3:
        raise ValueError(f"{path}: expected joints_m of shape (F, {N_JOINTS}, 3), got {j.shape}")
    return Lift3D(j, float(z["fps"]), z["up"], str(z["lifter"]))
