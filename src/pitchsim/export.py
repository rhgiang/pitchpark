"""Viewer export (M4): a pitcher's held-out at-bats as JSON for the Three.js viewer.

Python is the source of truth; the browser only interpolates the samples written here.
Layout (schema 2: schema 1 from the M4 plan, Task 3, plus Task 11's no-spin path and bases) is
read by every web module.

- "real" is Statcast's 9-parameter constant-acceleration fit, evaluated analytically from
  release to the front of the plate (not raw tracking points).
- "sim" is M1 + M3: physics.simulate from release_state with the pitch type's calibrated Cd /
  spin efficiency, that game's air density, and the per-pitch push, exactly as M3 scored it
  (seam.fit_pitch_types is the one place those fits live).
- "nospin" (schema 2) is a reference for seeing the movement: the same release state, the type's
  calibrated Cd, gravity, and nothing else (spin efficiency 0 removes the Magnus force; no M3
  push). Real plate minus no-spin plate is the "break vs no spin" (tunnel.break_x_in / _z_in):
  what spin and seams did to the pitch, as measured against a spinless ball thrown the same way.
- Held-out games only (validation.split_by_game, seed 0, inside fit_pitch_types).
- Raw rows give the at-bat skeleton so no pitch or at-bat is dropped: a row with no usable
  tracking is "no_tracking" (no paths), a pitch type M3 skipped is "no_model" (real path only).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pitchsim.physics import PhysicsParams, simulate
from pitchsim.seam import fit_pitch_types, predict_push
from pitchsim.statcast import _time_to_y, clean, fit_plate_crossing, initial_state, release_state
from pitchsim.tunneling import (DECISION_Y_FT, approach_angles_deg, perceived_velocity_mph,
                                separation_in)

SCHEMA = 2


def _num(x) -> float | None:
    """Float rounded to 1e-4 ft/s/..., or None for NaN / missing (JSON has no NaN)."""
    if x is None or pd.isna(x):
        return None
    return round(float(x), 4)


def _val(x):
    """JSON-safe scalar from a DataFrame cell: None for NaN, Python int/str otherwise."""
    if x is None or (not isinstance(x, str) and pd.isna(x)):
        return None
    return x.item() if hasattr(x, "item") else x


def _grid(t_end_s: float, dt_s: float) -> np.ndarray:
    """0, dt, 2dt, ... strictly before t_end, then t_end exactly (the plate)."""
    return np.append(np.arange(0.0, t_end_s - 1e-9, dt_s), t_end_s)


def _path(t: np.ndarray, pos: np.ndarray) -> dict:
    return {"t": np.round(t, 4).tolist(), "p": np.round(pos, 4).tolist()}


def _sampled(tr: dict, dt_s: float) -> tuple[dict, list, list]:
    """A physics.simulate result resampled on the viewer's grid: (path, plate [x, z, t], check).

    check: [t, x, y, z] at 1/4, 1/2, 3/4 of the flight from the 1 ms RK4 output itself, for the
    viewer's truth test (independent of the 5 ms samples). Times are rounded before evaluating,
    like check_real: the JSON stores t to 1e-4 s, and at ~130 ft/s 5e-5 s is ~0.08 in.
    """
    s = _grid(tr["t"][-1], dt_s)
    p = np.column_stack([np.interp(s, tr["t"], tr["pos"][:, k]) for k in range(3)])
    q = np.round(s[-1] * np.array([0.25, 0.5, 0.75]), 4)
    check = np.column_stack([q] + [np.interp(q, tr["t"], tr["pos"][:, k]) for k in range(3)])
    return _path(s, p), [_num(p[-1, 0]), _num(p[-1, 2]), _num(s[-1])], np.round(check, 4).tolist()


def export_pitcher(raw: pd.DataFrame, rho: pd.Series, dt_s: float = 0.005,
                   height_in: int | None = None) -> dict:
    """One pitcher's held-out at-bats in the viewer's JSON layout (schema 2).

    raw: this pitcher's Statcast rows, uncleaned (rows with no tracking still count).
    rho: air density (lb/ft^3) per row of raw (same index).
    dt_s: path sampling step in seconds; each path also ends exactly at y = 17/12 ft.
    height_in: the pitcher's height (StatsAPI) for the 3D delivery's scale; None if unknown.
    Added to schema 2 as an optional field, so readers of older files still work.
    """
    # copy() consolidates read_csv's ~120 column blocks (else pandas warns on assign).
    raw = raw.reset_index(drop=True).copy().assign(rho=np.asarray(rho), _raw=lambda d: d.index)
    cleaned = clean(raw)
    _, test, fits, _ = fit_pitch_types(cleaned, cleaned["rho"])
    test = test.reset_index(drop=True)
    held = raw[raw["game_pk"].isin(test["game_pk"].unique())].sort_values(
        ["game_date", "game_pk", "at_bat_number", "pitch_number"], kind="stable")

    # Real paths: the fit evaluated analytically. Times are since release: the fit's own
    # clock starts at y = 50 ft, so release is at t_rel < 0 and the plate at t_plate.
    pos50, vel50 = initial_state(test)
    acc = test[["ax", "ay", "az"]].to_numpy()
    plate_pos, plate_vel, t_plate = fit_plate_crossing(test)
    t_rel = _time_to_y(test["release_pos_y"].to_numpy(), test["vy0"].to_numpy(), acc[:, 1])
    flight = t_plate - t_rel
    # Approach angles from the Statcast-fit velocity at the plate (the "real" pitch).
    vaa, haa = approach_angles_deg(plate_vel)
    rel_pos, rel_vel = release_state(test)
    push = np.full((len(test), 2), np.nan)
    for ptype, tf in fits.items():
        m = (test["pitch_type"] == ptype).to_numpy()
        if m.any():
            push[m] = predict_push(tf.per, test[m])
    row_of = dict(zip(test["_raw"], test.index))

    def fit_at(i: int, s: np.ndarray) -> np.ndarray:
        tau = (t_rel[i] + s)[:, None]
        return pos50[i] + vel50[i] * tau + 0.5 * acc[i] * tau ** 2

    at_bats = []
    for (game_pk, ab_no), ab in held.groupby(["game_pk", "at_bat_number"], sort=False):
        first, pitches, prev = ab.iloc[0], [], None
        for _, r in ab.iterrows():
            i = row_of.get(r["_raw"])
            ptype = _val(r["pitch_type"])
            status = "no_tracking" if i is None else ("ok" if ptype in fits else "no_model")
            real = sim = plate_real = plate_sim = check = None
            nospin = plate_nospin = check_nospin = None
            path = None
            if i is not None:
                s = _grid(flight[i], dt_s)
                p = fit_at(i, s)
                p[-1] = plate_pos[i]    # same formula; exact at the plate by construction
                path = (s, p)
                real = _path(s, p)
                plate_real = [_num(plate_pos[i, 0]), _num(plate_pos[i, 2]), _num(flight[i])]
                # Round the times before evaluating: the JSON stores t to 1e-4 s, and a ball
                # at ~130 ft/s moves ~0.08 in in 5e-5 s, so positions must match the stored t.
                q = np.round(flight[i] * np.array([0.25, 0.5, 0.75]), 4)
                # Independent of the samples (Task 4 checks the samples against these).
                check = np.round(np.column_stack([q, fit_at(i, q)]), 4).tolist()
            if status == "ok":
                tf = fits[ptype]
                spin = float(test.at[i, "release_spin_rate"])
                axis = float(test.at[i, "spin_axis"])
                rho_i = float(test.at[i, "rho"])
                tr = simulate(rel_pos[i], rel_vel[i], spin, axis,
                              PhysicsParams(cd=tf.cd, spin_efficiency=tf.eff, rho=rho_i),
                              extra_acc=push[i])
                sim, plate_sim, _ = _sampled(tr, dt_s)
                tr0 = simulate(rel_pos[i], rel_vel[i], spin, axis,
                               PhysicsParams(cd=tf.cd, spin_efficiency=0.0, rho=rho_i))
                nospin, plate_nospin, check_nospin = _sampled(tr0, dt_s)
            sep = None
            if path is not None and prev is not None:
                try:
                    sep = _num(separation_in(prev, path, DECISION_Y_FT))
                except ValueError:   # a path that doesn't span the decision point
                    sep = None
            pitches.append({
                "pitch_number": int(r["pitch_number"]), "balls": int(r["balls"]),
                "strikes": int(r["strikes"]), "pitch_type": ptype,
                "pitch_name": _val(r.get("pitch_name")), "status": status,
                "speed_mph": _num(r["release_speed"]), "spin_rpm": _num(r["release_spin_rate"]),
                "spin_axis_deg": _num(r["spin_axis"]), "arm_angle_deg": _num(r.get("arm_angle")),
                "extension_ft": _num(r["release_extension"]),
                "release": [_num(r[c]) for c in ("release_pos_x", "release_pos_y",
                                                 "release_pos_z")],
                "sz_top_ft": _num(r["sz_top"]), "sz_bot_ft": _num(r["sz_bot"]),
                "description": _val(r["description"]), "events": _val(r["events"]),
                "real": real, "sim": sim, "plate_real": plate_real, "plate_sim": plate_sim,
                "check_real": check, "nospin": nospin, "plate_nospin": plate_nospin,
                "check_nospin": check_nospin,
                # Runners on base before the pitch (Statcast on_1b/2b/3b hold the runner's id).
                "bases": [_val(r.get(c)) is not None for c in ("on_1b", "on_2b", "on_3b")],
                "tunnel": {"vaa_deg": None if i is None else _num(vaa[i]),
                           "haa_deg": None if i is None else _num(haa[i]),
                           "perceived_mph": _num(perceived_velocity_mph(
                               r["release_speed"], r["release_extension"])),
                           "sep_prev_in": sep,
                           "break_x_in": _brk(plate_real, plate_nospin, 0),
                           "break_z_in": _brk(plate_real, plate_nospin, 1)}})
            prev = path
        last_event = ab["events"].dropna()
        at_bats.append({
            "id": f"{game_pk}-{ab_no}", "game_pk": int(game_pk),
            "game_date": str(first["game_date"]), "inning": int(first["inning"]),
            "top": first["inning_topbot"] == "Top", "outs": int(first["outs_when_up"]),
            "stand": _val(first["stand"]), "batter": int(first["batter"]),
            "event": _val(last_event.iloc[-1]) if len(last_event) else None,
            "pitches": pitches})

    name = str(raw["player_name"].iloc[0])
    names = held.dropna(subset=["pitch_type"]).groupby("pitch_type")["pitch_name"].first()
    pitch_types = {
        k: {"name": _val(v), "modelled": k in fits,
            "cd": _num(fits[k].cd) if k in fits else None,
            "spin_eff": _num(fits[k].eff) if k in fits else None}
        for k, v in names.items()}
    return {
        "meta": {"schema": SCHEMA, "units": "ft, s, mph, rpm, deg",
                 "axes": "x catcher's right, y plate->pitcher, z up",
                 "real_label": "Statcast fit", "sim_label": "M1 + M3",
                 "decision_y_ft": DECISION_Y_FT, "synthetic": "Synthetic" in name},
        "pitcher": {"name": name, "id": int(raw["pitcher"].iloc[0]),
                    "throws": str(raw["p_throws"].iloc[0]), "height_in": height_in},
        "pitch_types": pitch_types,
        "at_bats": at_bats}


def _brk(real: list | None, nospin: list | None, k: int) -> float | None:
    """Real minus no-spin plate arrival along x (k=0) or z (k=1), in inches."""
    return None if nospin is None else _num((real[k] - nospin[k]) * 12)


def write_json(obj: dict, path: Path) -> None:
    """Compact JSON (no NaN allowed: missing values must already be None)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, separators=(",", ":"), allow_nan=False), encoding="utf-8")
