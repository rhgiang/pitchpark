"""M7 "Face the pitcher": aim a pitch at a plate target, bank aimed M2 pitches, re-aim cheaply.

M6 samples where a pitch crosses the plate; M2 supplies release point, speed and spin; M1 + M3
fly it. M2's own release angles are discarded (they model where he aims, which is M6's job), so
each bank pitch is aimed with `aim_velocity` at its type's mean location, and the browser later
shifts it to the sampled target with `reaim_knots` (checked by `reaim_error_in`).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from pitchsim.export import KNOT_S, _sampled
from pitchsim.generative import fit_pitch_type, generated_release_state, sample
from pitchsim.physics import FRONT_OF_PLATE_Y, PhysicsParams, simulate
from pitchsim.seam import predict_push
from pitchsim.validation import FT_PER_S_TO_MPH


def plate_crossing(tr: dict) -> tuple[float, float]:
    """(x, z) in ft where a physics.simulate result crosses y = 17/12 ft (its last point)."""
    return float(tr["pos"][-1, 0]), float(tr["pos"][-1, 2])


def _direction(h_deg: float, v_deg: float, speed_fps: float) -> np.ndarray:
    """Velocity (ft/s) of given speed with Statcast-style release angles (see generative)."""
    d = np.array([np.tan(np.radians(h_deg)), -1.0, np.tan(np.radians(v_deg))])
    return speed_fps * d / np.linalg.norm(d)


def aim_velocity(rel_pos_ft, speed_fps: float, spin_rpm: float, axis_deg: float,
                 params: PhysicsParams, push, target_xz_ft) -> np.ndarray:
    """Release velocity (ft/s, norm = speed_fps) whose M1 + M3 flight crosses y = 17/12 ft at target.

    Solves the two release angles (horizontal, vertical) with least squares on the plate miss;
    speed is held fixed, so only the direction changes. Start: the straight line to the target
    (drag, gravity and Magnus then need a ~1-3 deg correction). push: (x, z) ft/s^2 from M3.
    """
    pos = np.asarray(rel_pos_ft, float)
    tx, tz = target_xz_ft
    run = pos[1] - FRONT_OF_PLATE_Y
    guess = np.degrees(np.arctan([(tx - pos[0]) / run, (tz - pos[2]) / run]))

    def miss(a):
        tr = simulate(pos, _direction(a[0], a[1], speed_fps), spin_rpm, axis_deg, params,
                      extra_acc=np.asarray(push, float))
        x, z = plate_crossing(tr)
        return [(x - tx) * 12, (z - tz) * 12]       # inches

    sol = least_squares(miss, guess, xtol=1e-12, ftol=1e-12, gtol=1e-12, diff_step=1e-6)
    err = float(np.hypot(*miss(sol.x)))
    if err > 0.01:      # never let a non-converged pitch into the bank
        raise ValueError(f"aim_velocity did not converge: plate miss {err:.4f} in (> 0.01 in)")
    return _direction(sol.x[0], sol.x[1], speed_fps)


def reaim_knots(sim: dict, dx_ft: float, dz_ft: float) -> dict:
    """Shift Hermite knots so the plate end moves by exactly (dx, dz); release is untouched.

    A small release-angle change is, to first order, a shift that grows linearly with distance
    from release: u = (y0 - y) / (y0 - y_end) runs 0 -> 1, and x, z gain u * (dx, dz). The
    velocities gain (dx, dz) * (-vy) / (y0 - y_end) (d u / dt), so the Hermite tangents agree
    with the shifted positions. y and T are unchanged. Drag/Magnus second-order effects are what
    `reaim_error_in` measures.
    """
    rows = [list(r) for r in sim["s"]]
    y0, y1 = rows[0][1], rows[-1][1]
    if y0 == y1:
        raise ValueError("reaim_knots needs knots spanning a distance in y (y0 == y_end)")
    out = []
    for x, y, z, vx, vy, vz in rows:
        u = (y0 - y) / (y0 - y1)
        k = -vy / (y0 - y1)
        out.append([x + u * dx_ft, y, z + u * dz_ft, vx + k * dx_ft, vy, vz + k * dz_ft])
    return {"s": out, "T": sim["T"]}


def make_bank(train: pd.DataFrame, rho, fits: dict, loc_means: dict, n_per_type: int = 200,
              seed: int = 0) -> dict[str, list[dict]]:
    """Per pitch type, n M2 pitches aimed (M1 + M3) at that type's mean plate location.

    train: seam.fit_pitch_types training frame (rho and release angles present). rho: scalar or
    array, the training games' median air density (lb/ft^3). loc_means[type]: (x, z) ft.
    Entries: sim knots, plate [x, z, T], aim [x, z], release [x, y, z], speed_mph, spin_rpm,
    spin_axis_deg, extension_ft, push [x, z] (needed to re-simulate), break_in [x, z] (real
    plate minus a no-spin flight, as in the export). The sampled release seam features are
    release-only, so predict_push is valid on them.
    """
    rho_med = float(np.median(rho))
    bank = {}
    for ptype, tf in fits.items():
        if ptype not in loc_means:
            continue
        model = fit_pitch_type(train[train["pitch_type"] == ptype], mixture=True, seed=seed)
        gen = sample(model, n_per_type, seed)
        pos, _ = generated_release_state(gen)
        speed = gen["release_speed"].to_numpy() / FT_PER_S_TO_MPH
        push = predict_push(tf.per, gen)
        params = PhysicsParams(cd=tf.cd, spin_efficiency=tf.eff, rho=rho_med)
        p0 = PhysicsParams(cd=tf.cd, spin_efficiency=0.0, rho=rho_med)
        aim = np.asarray(loc_means[ptype], float)
        entries = []
        for i in range(len(gen)):
            spin, axis = float(gen["release_spin_rate"].iloc[i]), float(gen["spin_axis"].iloc[i])
            v = aim_velocity(pos[i], speed[i], spin, axis, params, push[i], aim)
            tr = simulate(pos[i], v, spin, axis, params, extra_acc=push[i])
            sim, plate, _ = _sampled(tr, KNOT_S)
            tr0 = simulate(pos[i], v, spin, axis, p0)
            _, plate0, _ = _sampled(tr0, KNOT_S)
            entries.append({
                "sim": sim, "plate": plate, "aim": aim.tolist(), "release": pos[i].tolist(),
                "speed_mph": float(gen["release_speed"].iloc[i]), "spin_rpm": spin,
                "spin_axis_deg": axis, "extension_ft": float(gen["release_extension"].iloc[i]),
                "push": push[i].tolist(),
                "break_in": [(plate[0] - plate0[0]) * 12, (plate[1] - plate0[1]) * 12]})
        bank[ptype] = entries
    return bank


def reaim_error_in(bank: dict, targets_by_type: dict, params_by_type: dict) -> pd.DataFrame:
    """Error (inches) of `reaim_knots` vs a true re-simulation aimed with `aim_velocity`.

    Pitch i of a type is re-aimed to targets_by_type[type][i % len] (x, z in ft). Columns:
    pitch_type, target_x_ft, target_z_ft, plate_in (distance at the plate end) and path_in
    (max distance over the knots, compared at equal y). The plate end of the shift is exact by
    construction, so path_in (mid-flight, where drag/Magnus act) is the informative number.
    """
    rows = []
    for ptype, entries in bank.items():
        tg = np.atleast_2d(targets_by_type[ptype])
        params = params_by_type[ptype]
        for i, e in enumerate(entries):
            t = tg[i % len(tg)]
            sim = reaim_knots(e["sim"], t[0] - e["aim"][0], t[1] - e["aim"][1])
            v = aim_velocity(e["release"], e["speed_mph"] / FT_PER_S_TO_MPH, e["spin_rpm"],
                             e["spin_axis_deg"], params, e["push"], t)
            tr = simulate(e["release"], v, e["spin_rpm"], e["spin_axis_deg"], params,
                          extra_acc=np.asarray(e["push"]))
            k = np.array(sim["s"])
            y = tr["pos"][::-1, 1]                       # y falls with time; interp needs rising
            tx = np.interp(k[:, 1], y, tr["pos"][::-1, 0])
            tz = np.interp(k[:, 1], y, tr["pos"][::-1, 2])
            d = np.hypot(k[:, 0] - tx, k[:, 2] - tz) * 12
            rows.append({"pitch_type": ptype, "target_x_ft": t[0], "target_z_ft": t[1],
                         "plate_in": float(d[-1]), "path_in": float(d.max())})
    return pd.DataFrame(rows)
