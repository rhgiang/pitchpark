"""
Make a fake pitcher whose physics we know exactly, in Statcast's CSV format.

    python -m pitchsim.synthetic          # writes data/synthetic.csv
    python scripts/validate.py data/synthetic.csv

Why: it tests the whole pipeline end to end without network access. Because
we choose the true Cd and spin efficiency, calibration should recover them;
if it doesn't, the bug is in our code, not in the data.

These are NOT real pitches. Use scripts/fetch.py for real ones.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from pitchsim.physics import FRONT_OF_PLATE_Y, STATCAST_Y0, PhysicsParams, simulate

TRUE_CD = 0.35
# name, code, mph, rpm, spin_axis, spin efficiency, share of pitches
ARSENAL = [
    ("4-Seam Fastball", "FF", 95.0, 2400, 210, 0.92, 0.45),
    ("Slider",          "SL", 86.0, 2550,  95, 0.40, 0.25),
    ("Changeup",        "CH", 87.0, 1800, 240, 0.85, 0.18),
    ("Curveball",       "CU", 79.0, 2700,  35, 0.75, 0.12),
]


def statcast_fit(t, pos):
    """Least-squares constant-acceleration fit, re-expressed at y = 50 ft,
    mimicking how Statcast summarizes a tracked trajectory."""
    coefs = [np.polyfit(t, pos[:, i], 2) for i in range(3)]   # a/2, v, p per axis
    ay2, vy, py = coefs[1]
    # time where fitted y == 50
    roots = np.roots([ay2, vy, py - STATCAST_Y0])
    t50 = roots[np.argmin(np.abs(roots - t[np.argmin(np.abs(pos[:, 1] - 50))]))].real
    v0 = [2 * c[0] * t50 + c[1] for c in coefs]
    a = [2 * c[0] for c in coefs]
    return v0, a


def make(n=1500, seed=0):
    rng = np.random.default_rng(seed)
    names = [a[0] for a in ARSENAL]
    shares = np.array([a[6] for a in ARSENAL])
    rows = []
    for _ in range(n):
        name, code, mph, rpm, axis, eff, _ = ARSENAL[rng.choice(len(ARSENAL), p=shares)]
        speed = (mph + rng.normal(0, 0.8)) / 0.681818
        spin = rpm + rng.normal(0, 60)
        sa = axis + rng.normal(0, 6)
        ext = 6.3 + rng.normal(0, 0.15)
        release = np.array([-1.9 + rng.normal(0, 0.1), 60.5 - ext, 5.9 + rng.normal(0, 0.1)])
        # aim roughly at the zone, with spread
        target = np.array([rng.normal(0, 0.8), FRONT_OF_PLATE_Y, rng.normal(2.5, 0.8)])
        d = target - release
        d[2] += 2.8   # aim above the target to account for drop
        vel = speed * d / np.linalg.norm(d)

        p = PhysicsParams(cd=TRUE_CD, spin_efficiency=eff)
        tr = simulate(release, vel, spin, sa, p)
        v0, a = statcast_fit(tr["t"], tr["pos"])
        end = tr["pos"][-1]
        rows.append({
            "player_name": "Synthetic, Pitcher",
            "game_pk": int(rng.integers(1, 41)),   # 40 fake games, for the by-game split
            "pitch_type": code, "pitch_name": name,
            "release_speed": speed * 0.681818,
            "release_pos_x": release[0], "release_pos_y": release[1],
            "release_pos_z": release[2], "release_extension": ext,
            "vx0": v0[0], "vy0": v0[1], "vz0": v0[2],
            "ax": a[0], "ay": a[1], "az": a[2],
            "plate_x": end[0], "plate_z": end[2],
            "release_spin_rate": spin, "spin_axis": sa % 360,
            "true_cd": TRUE_CD, "true_spin_eff": eff,
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    out = Path(__file__).resolve().parents[2] / "data"
    out.mkdir(exist_ok=True)
    df = make()
    df.to_csv(out / "synthetic.csv", index=False)
    print(f"Wrote {len(df)} synthetic pitches to {out / 'synthetic.csv'}")
    print("True values: Cd =", TRUE_CD, "| spin efficiency:",
          {a[1]: a[5] for a in ARSENAL})
