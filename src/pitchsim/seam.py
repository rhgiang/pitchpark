"""Seam-shifted wake (Milestone 3): learn the push Magnus can't explain.

In experiments the seams can trip the boundary layer on one side of the ball, shifting the wake
and adding a sideways/vertical force that Magnus + drag don't model (Barton Smith's work;
Driveline's seam-shifted-wake write-ups). That is the hypothesis, not something measured here:
M1 still misses some types by inches in a consistent direction, and this module fits that miss
whatever its cause (see results/M3.md).

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
from scipy.stats import ks_2samp
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pitchsim.generative import (MIN_TEST, MIN_TRAIN, add_release_angles, circular_mean_deg,
                                 fit_pitch_type, generated_release_state, movement_in, sample)
from pitchsim.physics import simulate_batch
from pitchsim.statcast import (_time_to_y, fit_plate_crossing, observed_break_direction_deg,
                               release_state)
from pitchsim.validation import (EFF_MIN, MAX_TRAIN, angle_diff, calibrate, residuals, run_sim,
                                 split_by_game)

SEAM_FEATURES = ["sin_axis", "cos_axis", "release_spin_rate", "release_speed",
                 "release_pos_x", "release_pos_z", "release_extension"]
FORBIDDEN = ["ax", "ay", "az", "vx0", "vy0", "vz0", "pfx_x", "pfx_z", "plate_x", "plate_z"]
_PLAIN = SEAM_FEATURES[2:]


def seam_design(df: pd.DataFrame) -> np.ndarray:
    """N x 7 feature matrix in SEAM_FEATURES order, from release-only columns."""
    axis = np.radians(df["spin_axis"].to_numpy(float))
    return np.column_stack([np.sin(axis), np.cos(axis), df[_PLAIN].to_numpy(float)])


def push_target(df: pd.DataFrame, cd: float | np.ndarray, eff: float | np.ndarray) -> np.ndarray:
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
    """A fitted push predictor for one pitch type (push in ft/s^2)."""
    kind: str                      # "fixed", "per_pitch" or "fixed_fallback"
    mean: np.ndarray               # (2,) mean push x, z in ft/s^2 (training target)
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


N_BOOT = 1000
# Smallest drop that counts. A synthetic pitcher with NO seam force on a type still drops by up
# to 0.6 in, because the push also removes a constant M1 offset (probably the release-state
# artefact above). Synthetic has no measurement noise, so this is an estimate for real data.
MIN_DROP_IN = 0.6
_RAW_FEATURES = ["spin_axis"] + _PLAIN


def median_drop_ci(before_in: np.ndarray, after_in: np.ndarray, games: np.ndarray,
                   seed: int = 0, n_boot: int = N_BOOT) -> tuple[float, float, float]:
    """(median(before) - median(after), 2.5%, 97.5%), bootstrapping held-out GAMES.

    Pitches in a game share air and mound, the same reason the split is by game, so the
    resampling unit is the game: a drawn game brings all its pitches, paired before/after.
    """
    before, after, games = (np.asarray(a) for a in (before_in, after_in, games))
    groups = [np.flatnonzero(games == g) for g in np.unique(games)]
    rng = np.random.default_rng(seed)
    drops = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.concatenate([groups[i] for i in rng.integers(len(groups), size=len(groups))])
        drops[b] = np.median(before[idx]) - np.median(after[idx])
    lo, hi = np.percentile(drops, [2.5, 97.5])
    return float(np.median(before) - np.median(after)), float(lo), float(hi)


def grade(n_test: int, n_games_test: int, drop: tuple[float, float, float]
          ) -> tuple[bool, bool | object]:
    """(graded, pass) for one pitch type; pass is pd.NA when not graded.

    Graded needs MIN_TEST held-out pitches and at least 2 held-out games: with one game every
    bootstrap draw is that game, so the interval collapses to a point and says nothing.
    Pass needs the drop (in) to reach MIN_DROP_IN with its 95% interval above zero.
    """
    if n_test < MIN_TEST or n_games_test < 2:
        return False, pd.NA
    return True, bool(drop[0] >= MIN_DROP_IN and drop[1] > 0)


def evaluate_pitcher(df: pd.DataFrame, rho: np.ndarray, seed: int = 0, eff_min: float = EFF_MIN,
                     split_seed: int = 0
                     ) -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict[str, pd.DataFrame]]:
    """Held-out score of M1, M1 + fixed push, M1 + per-pitch push, per pitch type.

    rho is air density (lb/ft^3) per row of df. eff_min is M1's lower efficiency bound (see
    calibrate). split_seed picks the train/held-out game split (default 0, the one every
    result uses; scripts/seam_split_check.py varies it). Returns (summary, knock_on, skipped,
    clouds).
    summary: median held-out miss (in) for each variant on the same pitches, M3 biases, game
    bootstrap intervals for the drop (M1 -> per-pitch) and for per-pitch vs fixed, the training
    push and spin-axis gap, and the pass rule (see grade). knock_on (reported, not graded):
    M2-generated pitches flown with M1 and with M1 + per-pitch push, KS vs held-out Statcast
    movement.
    clouds[pitch_type]: one row per held-out pitch, signed miss (in) for M1 only and for M1 +
    per-pitch push (dx_m1_in, dz_m1_in, dx_pp_in, dz_pp_in).
    """
    df = add_release_angles(df.assign(rho=np.asarray(rho))).reset_index(drop=True)
    skipped = []
    for c in _RAW_FEATURES:
        k = int(df[c].isna().sum())
        if k:
            skipped.append(f"dropped {k} rows with missing {c}")
    df = df.dropna(subset=_RAW_FEATURES)
    train_all, test_all = split_by_game(df, seed=split_seed)

    rows, knock, clouds = [], [], {}
    for ptype, g in df.groupby("pitch_type"):
        train = train_all[train_all["pitch_type"] == ptype]
        test = test_all[test_all["pitch_type"] == ptype]
        if len(train) < MIN_TRAIN:
            skipped.append(f"{ptype} (n={len(g)}): fewer than {MIN_TRAIN} training pitches")
            continue
        if len(test) == 0:
            skipped.append(f"{ptype} (n={len(g)}): no pitches in held-out games")
            continue
        # Calibrate exactly as M1/M2 (capped sample); the push is fitted on ALL training
        # pitches with those Cd/eff.
        cal = train.sample(MAX_TRAIN, random_state=seed) if len(train) > MAX_TRAIN else train
        cd, eff = calibrate(cal, "release", eff_min)
        target = push_target(train, cd, eff)
        fixed, per = fit_fixed(target), fit_per_pitch(train, target, seed)

        fit_pos, fit_vel, t_s = fit_plate_crossing(test)
        miss, bias, signed = {}, {}, {}
        for name, acc in (("m1", None), ("fixed", predict_push(fixed, test)),
                          ("pp", predict_push(per, test))):
            dx, dz, _ = residuals(test, fit_pos, fit_vel, cd, eff, "release", extra_acc=acc)
            miss[name], bias[name] = np.hypot(dx, dz), (dx.mean(), dz.mean())
            signed[f"dx_{name}_in"], signed[f"dz_{name}_in"] = dx, dz
        clouds[ptype] = pd.DataFrame({k: signed[k] for k in
                                      ("dx_m1_in", "dz_m1_in", "dx_pp_in", "dz_pp_in")})
        games = test["game_pk"].to_numpy()
        drop = median_drop_ci(miss["m1"], miss["pp"], games, seed)
        vs_fixed = median_drop_ci(miss["fixed"], miss["pp"], games, seed)
        n_games_test = int(test["game_pk"].nunique())
        graded, passed = grade(len(test), n_games_test, drop)
        rows.append({
            "pitch_type": ptype,
            "pitch_name": g["pitch_name"].iloc[0] if "pitch_name" in g else ptype,
            "n_train": len(train), "n_test": len(test), "n_games_test": n_games_test,
            "n_games_train": int(train["game_pk"].nunique()),  # fixed.n_games is 0 = unknown
            "cd": cd, "spin_eff": eff,
            "err_m1_in": np.median(miss["m1"]), "err_fixed_in": np.median(miss["fixed"]),
            "err_per_pitch_in": np.median(miss["pp"]),
            "bias_x_m1_in": bias["m1"][0], "bias_z_m1_in": bias["m1"][1],
            "bias_x_fixed_in": bias["fixed"][0], "bias_z_fixed_in": bias["fixed"][1],
            "bias_x_m3_in": bias["pp"][0], "bias_z_m3_in": bias["pp"][1],
            "drop_in": drop[0], "drop_lo_in": drop[1], "drop_hi_in": drop[2],
            "pp_vs_fixed_in": vs_fixed[0], "pp_vs_fixed_lo_in": vs_fixed[1],
            "pp_vs_fixed_hi_in": vs_fixed[2],
            "push_x_fts2": fixed.mean[0], "push_z_fts2": fixed.mean[1],
            # measured minus movement-implied axis, signed, in [-180, 180)
            "axis_gap_deg": float(angle_diff(circular_mean_deg(angle_diff(
                train["spin_axis"].to_numpy(float), observed_break_direction_deg(train))), 0.0)),
            "alpha": np.nan if per.alpha is None else per.alpha, "model_kind": per.kind,
            "graded": graded, "pass": passed})

        # M2 knock-on: generated pitches through M1 and M1 + per-pitch push vs Statcast movement
        # (same cloud and formula as generative.evaluate_pitcher).
        _, v_rel = release_state(test)
        t_flight = t_s - _time_to_y(test["release_pos_y"].to_numpy(), test["vy0"].to_numpy(),
                                    test["ay"].to_numpy())
        stat = movement_in(v_rel, fit_vel, t_flight)
        # large draw so the generated side isn't noisy
        gen = sample(fit_pitch_type(train, mixture=False, seed=seed), 1000, seed)
        pos_g, v_g = generated_release_state(gen)
        for variant, acc in (("m1", None), ("m1+m3", predict_push(per, gen))):
            _, vp, t = simulate_batch(pos_g, v_g, gen["release_spin_rate"].to_numpy(),
                                      gen["spin_axis"].to_numpy(), cd, eff,
                                      # training-median rho: a generated pitch has no game
                                      rho=float(train["rho"].median()), extra_acc=acc)
            for comp, sim, ref in zip("xz", movement_in(v_g, vp, t), stat):
                r = ks_2samp(sim, ref)
                knock.append({"pitch_type": ptype, "variant": variant, "component": comp,
                              "ks_stat": r.statistic, "p_value": r.pvalue,
                              "mean_diff_in": float(sim.mean() - ref.mean())})
    cols = ["pitch_type", "variant", "component", "ks_stat", "p_value", "mean_diff_in"]
    return pd.DataFrame(rows), pd.DataFrame(knock, columns=cols), skipped, clouds
