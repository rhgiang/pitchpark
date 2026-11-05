"""
M3 controls on SYNTHETIC pitches (make(n=1500, seed=0), constant air). Fake data, known truth.

    python scripts/seam_controls.py

Answers questions the real runs can't, and writes results/synthetic/m3_controls.csv:
  no_push   What does M3 "find" when there is no seam force at all? (source of the 0.6 in floor)
  artefact  How wrong is release_state, and how big a push does that alone look like?
            Uses the true Cd and efficiency, so nothing but the start state is off.
  along / off_axis  A planted CH push along the spin force vs off it: what can M3 see?
"""

from pathlib import Path

import numpy as np
import pandas as pd

import pitchsim.synthetic as syn
from pitchsim.physics import air_density
from pitchsim.seam import evaluate_pitcher, push_target
from pitchsim.statcast import fit_plate_crossing, release_state
from pitchsim.validation import residuals

OUT = Path(__file__).resolve().parents[1] / "results" / "synthetic" / "m3_controls.csv"
COLS = ["pitch_type", "n_test", "n_games_test", "spin_eff", "err_m1_in", "err_fixed_in",
        "err_per_pitch_in", "drop_in", "drop_lo_in", "drop_hi_in", "pp_vs_fixed_in",
        "push_x_fts2", "push_z_fts2", "graded", "pass"]
ALONG = lambda c, a: (5.0, -4.0) if c == "CH" else (0.0, 0.0)
OFF_AXIS = lambda c, a: (4.0, 5.0) if c == "CH" else (0.0, 0.0)


def make_recording_velocity() -> tuple[pd.DataFrame, np.ndarray]:
    """make() with no push, plus the true release velocity of each pitch (N x 3, ft/s).
    make() doesn't store it, so listen in on its simulate() calls."""
    seen, real = [], syn.simulate

    def spy(release, vel, *a, **k):
        seen.append(vel)
        return real(release, vel, *a, **k)

    syn.simulate = spy
    try:
        return syn.make(n=1500, seed=0), np.array(seen)
    finally:
        syn.simulate = real


def main():
    rho = air_density(70.0, 0.0)
    df, true_vel = make_recording_velocity()
    parts = []
    for name, frame in (("no_push", df), ("along", syn.make(n=1500, seed=0, push=ALONG)),
                        ("off_axis", syn.make(n=1500, seed=0, push=OFF_AXIS))):
        summary = evaluate_pitcher(frame, np.full(len(frame), rho))[0]
        parts.append(summary[COLS].assign(control=name))

    df = df.assign(rho=rho)
    cd, eff = df["true_cd"].to_numpy(), df["true_spin_eff"].to_numpy()
    vel_err = np.abs(release_state(df)[1] - true_vel)
    fit_pos, fit_vel, _ = fit_plate_crossing(df)
    dx, dz, _ = residuals(df, fit_pos, fit_vel, cd, eff, "release")
    miss = np.hypot(dx, dz)
    target = push_target(df, cd, eff)
    art = pd.DataFrame({"pitch_type": df["pitch_type"], "floor_miss_in": miss,
                        "push_x_fts2": target[:, 0], "push_z_fts2": target[:, 1]})
    parts.append(art.groupby("pitch_type").agg(
        floor_miss_in=("floor_miss_in", "median"), push_x_fts2=("push_x_fts2", "mean"),
        push_z_fts2=("push_z_fts2", "mean")).reset_index().assign(control="artefact"))
    parts.append(pd.DataFrame([{
        "pitch_type": "ALL", "control": "artefact", "floor_miss_in": np.median(miss),
        "vel_err_x_fts": np.median(vel_err[:, 0]), "vel_err_y_fts": np.median(vel_err[:, 1]),
        "vel_err_z_fts": np.median(vel_err[:, 2])}]))
    out = pd.concat(parts, ignore_index=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.round(3).to_csv(OUT, index=False)
    print("SYNTHETIC controls")
    print(out.round(3).to_string(index=False))
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
