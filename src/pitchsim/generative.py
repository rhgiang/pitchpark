"""
Per-pitch-type generative model of how a pitch leaves the hand (M2).

One Gaussian (or small Gaussian mixture) per pitch type over release position,
extension, speed, spin rate, spin axis and the two release angles.
"""

import copy
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, multivariate_normal
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from pitchsim.physics import G, simulate_batch
from pitchsim.statcast import _time_to_y, fit_plate_crossing, release_state
from pitchsim.validation import FT_PER_S_TO_MPH, MAX_TRAIN, calibrate, split_by_game

# Features scored against real pitches.
GRADED = ["release_pos_x", "release_pos_z", "release_extension",
          "release_speed", "release_spin_rate", "spin_axis"]
# Release angles are modelled but not graded: they are where he aims, which is M6's job.
# They are still needed so the sampled velocity direction is consistent with the release point.
ANGLES = ["release_angle_v_deg", "release_angle_h_deg"]
FEATURES = GRADED + ANGLES

# Statcast publishes these columns rounded; the generator makes continuous values. A classifier
# can tell real from generated on the decimals alone (Misiorowski FF: AUC 0.997 unrounded). So
# the evaluation rounds generated pitches like the measurement does; the generator is unchanged.
STATCAST_DECIMALS = {"release_pos_x": 2, "release_pos_z": 2, "release_extension": 1,
                     "release_speed": 1, "release_spin_rate": 0, "spin_axis": 0}


def round_like_statcast(gen: pd.DataFrame) -> pd.DataFrame:
    """Copy of gen with the graded columns rounded to Statcast's precision (axis stays in [0, 360))."""
    out = gen.copy()
    for c, d in STATCAST_DECIMALS.items():
        out[c] = out[c].round(d)
    out["spin_axis"] = out["spin_axis"] % 360  # 359.6 rounds to 360 -> 0
    return out

# Too few pitches and an 8-D covariance (36 parameters) is mostly noise.
MIN_TRAIN = 30
MIN_TEST = 30
# A mixture needs enough data per component to be worth its extra parameters.
MIN_MIXTURE = 100
MAX_COMPONENTS = 4
# Number of generated sets averaged in the classifier test, and the AUC counted as a pass.
N_DRAWS = 10
PASS_AUC = 0.55
N_BOOT = 200


def add_release_angles(df: pd.DataFrame) -> pd.DataFrame:
    """Copy of df with vertical / horizontal release angles (degrees) from release velocity.

    y points toward the pitcher, so the ball travels along -vy: the vertical angle is
    atan2(vz, -vy) (negative = thrown downward), the horizontal one atan2(vx, -vy).
    """
    _, vel = release_state(df)
    out = df.copy()
    out["release_angle_v_deg"] = np.degrees(np.arctan2(vel[:, 2], -vel[:, 1]))
    out["release_angle_h_deg"] = np.degrees(np.arctan2(vel[:, 0], -vel[:, 1]))
    return out


def circular_mean_deg(a: np.ndarray) -> float:
    """Mean direction of angles in degrees, in [0, 360)."""
    r = np.radians(np.asarray(a, dtype=float))
    return float(np.degrees(np.arctan2(np.sin(r).mean(), np.cos(r).mean())) % 360)


def unwrap_deg(a: np.ndarray, center: float) -> np.ndarray:
    """Shift angles (degrees) by multiples of 360 into center +/- 180.

    Spin axis is circular, so 359 and 1 are 2 deg apart. A plain mean of them is 180
    and the variance is huge; unwrapping around a circular mean makes a Gaussian fit.
    """
    return (np.asarray(a, dtype=float) - center + 180) % 360 - 180 + center


@dataclass
class PitchTypeModel:
    gmm: GaussianMixture
    axis_center_deg: float  # spin_axis was unwrapped around this before fitting
    n_train: int


def fit_pitch_type(train: pd.DataFrame, mixture: bool, seed: int = 0) -> PitchTypeModel:
    """Fit a Gaussian (mixture=False) or BIC-chosen mixture to one pitch type's FEATURES."""
    X = train[FEATURES].to_numpy(dtype=float).copy()
    center = circular_mean_deg(X[:, FEATURES.index("spin_axis")])
    X[:, FEATURES.index("spin_axis")] = unwrap_deg(X[:, FEATURES.index("spin_axis")], center)

    # Full covariance: release height and extension (and speed / spin) move together,
    # so a diagonal model would sample combinations he never throws.
    def fit(k: int) -> GaussianMixture:
        return GaussianMixture(k, covariance_type="full", random_state=seed).fit(X)

    if mixture and len(X) >= MIN_MIXTURE:
        # BIC penalises each extra component's parameters, so it stops adding
        # components once they only fit noise (log-likelihood alone always improves).
        gmm = min((fit(k) for k in range(1, MAX_COMPONENTS + 1)), key=lambda g: g.bic(X))
    else:
        gmm = fit(1)
    return PitchTypeModel(gmm, center, len(X))


def sample(model: PitchTypeModel, n: int, seed: int) -> pd.DataFrame:
    """Draw n synthetic pitches (columns FEATURES); spin_axis wrapped back to [0, 360)."""
    g = copy.copy(model.gmm)  # sklearn draws with its own random_state; don't mutate the model
    g.random_state = seed
    out = pd.DataFrame(g.sample(n)[0], columns=FEATURES)
    out["spin_axis"] = out["spin_axis"] % 360
    return out


def _design(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    """Feature matrix with spin_axis replaced by sin and cos of the angle.

    Raw axis jumps from 360 to 0, so a classifier could "detect" fakes by that
    artificial gap; sin/cos puts 359 and 1 next to each other.
    """
    parts = []
    for c in cols:
        if c == "spin_axis":
            r = np.radians(df[c].to_numpy(dtype=float))
            parts += [np.sin(r), np.cos(r)]
        else:
            parts.append(df[c].to_numpy(dtype=float))
    return np.column_stack(parts)


def classifier_auc(real: pd.DataFrame, generated: list[pd.DataFrame], cols: list[str],
                   center: float, seed: int = 0) -> tuple[float, float, float]:
    """Classifier two-sample test: can a model tell real (0) from generated (1)?

    Returns (mean AUC, mean 2.5% bound, mean 97.5% bound) over the generated sets;
    AUC near 0.5 means indistinguishable. Each set should be the same size as `real`.
    `center` is unused here (sin/cos needs no unwrapping); it is kept so the signature
    matches marginal_ks.

    Why a nonlinear classifier: a fitted Gaussian matches means and linear correlations
    by construction, so logistic regression would pass it trivially. Gradient boosting
    can find what is left (curved or clumped structure). Out-of-fold predictions keep
    the classifier from scoring itself on pitches it trained on.
    Why average several sets: one draw of a few dozen pitches gives a noisy AUC.
    Ref: Lopez-Paz & Oquab, "Revisiting Classifier Two-Sample Tests" (ICLR 2017).
    """
    rng = np.random.default_rng(seed)
    Xr = _design(real, cols)
    aucs, los, his = [], [], []
    for gen in generated:
        X = np.vstack([Xr, _design(gen, cols)])
        y = np.r_[np.zeros(len(Xr)), np.ones(len(gen))].astype(int)
        p = cross_val_predict(HistGradientBoostingClassifier(random_state=seed), X, y,
                              cv=StratifiedKFold(5, shuffle=True, random_state=seed),
                              method="predict_proba")[:, 1]
        aucs.append(roc_auc_score(y, p))
        # Bootstrap the predictions to see how much the AUC could move by chance.
        boot = []
        for _ in range(N_BOOT):
            i = rng.integers(0, len(y), len(y))
            if len(np.unique(y[i])) == 2:
                boot.append(roc_auc_score(y[i], p[i]))
        los.append(np.percentile(boot, 2.5)); his.append(np.percentile(boot, 97.5))
    return float(np.mean(aucs)), float(np.mean(los)), float(np.mean(his))


def marginal_ks(real: pd.DataFrame, generated: pd.DataFrame, center: float) -> pd.DataFrame:
    """Per-feature two-sample KS test of GRADED columns; flagged when p < 0.01.

    Spin axis is unwrapped around `center` first so a cluster straddling 0/360
    is not read as two distant groups.
    """
    rows = []
    for c in GRADED:
        a, b = real[c].to_numpy(dtype=float), generated[c].to_numpy(dtype=float)
        if c == "spin_axis":
            a, b = unwrap_deg(a, center), unwrap_deg(b, center)
        r = ks_2samp(a, b)
        rows.append({"feature": c, "ks_stat": r.statistic, "p_value": r.pvalue,
                     "flagged": bool(r.pvalue < 0.01)})
    return pd.DataFrame(rows)


def movement_in(vel_release: np.ndarray, vel_plate: np.ndarray,
                t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Horizontal and vertical break (inches) from release and plate velocities (ft/s), t in s.

    The average non-gravity acceleration over the flight is (v_plate - v_release) / t, with
    gravity added back to z. Break is 0.5 * a * t^2: how far that acceleration alone moves the
    ball. One formula for every cloud, so differences between clouds are not formula artefacts.
    """
    a = (vel_plate - vel_release) / t[:, None]
    a_z = a[:, 2] + G
    return 0.5 * a[:, 0] * t ** 2 * 12, 0.5 * a_z * t ** 2 * 12


def generated_release_state(gen: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Release position (ft) and velocity (ft/s) of generated pitches (inverse of the angles).

    The angles satisfy vx / -vy = tan(h) and vz / -vy = tan(v), so the direction is
    (tan h, -1, tan v); its length is set by release_speed.
    """
    pos = np.column_stack([gen["release_pos_x"], 60.5 - gen["release_extension"],
                           gen["release_pos_z"]])
    d = np.column_stack([np.tan(np.radians(gen["release_angle_h_deg"])),
                         -np.ones(len(gen)),
                         np.tan(np.radians(gen["release_angle_v_deg"]))])
    speed = gen["release_speed"].to_numpy() / FT_PER_S_TO_MPH
    return pos, speed[:, None] * d / np.linalg.norm(d, axis=1, keepdims=True)

# Context that can shift the cloud: fatigue, times through the order, count.
CONTEXT = ["pitch_count", "n_thruorder_pitcher", "balls", "strikes"]


def pitch_count_in_game(df: pd.DataFrame) -> pd.Series:
    """1-based pitch number within each game, ordered by at_bat_number then pitch_number.

    Run on the full CSV (all pitch types, before filtering): fatigue depends on every pitch
    thrown, not just those of one type. Aligned to df.index.
    """
    order = df.sort_values(["game_pk", "at_bat_number", "pitch_number"])
    return (order.groupby("game_pk").cumcount() + 1).reindex(df.index)


@dataclass
class ContextModel:
    coef: np.ndarray  # (1 + len(CONTEXT), len(GRADED)); first row is the intercept
    cov: np.ndarray   # shared residual covariance of GRADED
    axis_center_deg: float


def _graded_matrix(df: pd.DataFrame, center: float) -> np.ndarray:
    """GRADED columns as floats, spin axis unwrapped around center."""
    Y = df[GRADED].to_numpy(dtype=float).copy()
    i = GRADED.index("spin_axis")
    Y[:, i] = unwrap_deg(Y[:, i], center)
    return Y


def _context_design(df: pd.DataFrame) -> np.ndarray:
    return np.column_stack([np.ones(len(df)), df[CONTEXT].to_numpy(dtype=float)])


def fit_context(train: pd.DataFrame) -> ContextModel:
    """Least-squares fit of GRADED on CONTEXT: mean = [1, context] @ coef, shared covariance.

    Linear-in-context with one shared covariance is the simplest model where context shifts
    the cloud: multivariate linear regression with Gaussian residuals. Only GRADED is modelled;
    the release angles encode aim, which depends on count and is M6's job.
    The covariance divides by n (maximum likelihood), so it is like-for-like with the
    intercept-only baseline in context_gain.
    """
    center = circular_mean_deg(train["spin_axis"].to_numpy())
    X, Y = _context_design(train), _graded_matrix(train, center)
    coef = np.linalg.lstsq(X, Y, rcond=None)[0]
    resid = Y - X @ coef
    return ContextModel(coef, resid.T @ resid / len(Y), center)


def sample_context(model: ContextModel, context: pd.DataFrame, seed: int) -> pd.DataFrame:
    """One draw of GRADED per context row; spin_axis wrapped back to [0, 360)."""
    rng = np.random.default_rng(seed)
    mean = _context_design(context) @ model.coef
    out = pd.DataFrame(mean + rng.multivariate_normal(np.zeros(len(GRADED)), model.cov,
                                                      len(context)), columns=GRADED)
    out["spin_axis"] = out["spin_axis"] % 360
    return out


def context_gain(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """Held-out mean log-likelihood per pitch of the context model vs an intercept-only one.

    Log-likelihood scores the whole distribution (mean and spread), not just the mean.
    Rows missing any CONTEXT value are dropped from both sets; n_dropped counts test rows.
    Below MIN_TRAIN usable training rows the log-likelihoods are NaN.
    """
    tr, te = train.dropna(subset=CONTEXT), test.dropna(subset=CONTEXT)
    r = {"n_train": len(tr), "n_test": len(te), "n_dropped": len(test) - len(te),
         "ll_base": np.nan, "ll_context": np.nan}
    if len(tr) < MIN_TRAIN or len(te) == 0:
        return r
    m = fit_context(tr)
    Yte = _graded_matrix(te, m.axis_center_deg)
    base = fit_context(tr.assign(**{c: 0.0 for c in CONTEXT}))  # zero columns -> intercept only
    # lstsq gives the min-norm solution, so the zeroed columns get coef 0: a pure mean fit.
    Xb = _context_design(te.assign(**{c: 0.0 for c in CONTEXT}))
    r["ll_base"] = float(multivariate_normal(np.zeros(len(GRADED)), base.cov).logpdf(
        Yte - Xb @ base.coef).mean())
    r["ll_context"] = float(multivariate_normal(np.zeros(len(GRADED)), m.cov).logpdf(
        Yte - _context_design(te) @ m.coef).mean())
    return r


def evaluate_pitcher(df: pd.DataFrame, rho: np.ndarray, seed: int = 0
                     ) -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict, pd.DataFrame]:
    """Score the generative model on held-out games, per pitch type and per model.

    rho is air density (lb/ft^3) per row of df. Returns (summary, ks, skipped, clouds, context):
    summary has one row per pitch type x model ("gaussian", "mixture") with the classifier
    AUC on GRADED features (auc, bounds) and on all FEATURES (auc8); ks has the marginal KS
    rows; clouds[pitch_type] holds held-out pitches, 1,000 generated ones, and break (in)
    from three sources: Statcast's fit (a), the M1 simulation of the real release states (b),
    and the M1 simulation of generated release states (c).

    context has one row per fitted pitch type (see context_gain; ll_gain = ll_context - ll_base);
    it is empty, with one skipped note, if a CONTEXT source column is missing.

    Movement is not graded here: its error belongs to M1 / M3 (physics). Comparing (b) with
    (c) isolates M2's error, because both go through the same simulator and calibration.
    """
    df = add_release_angles(df.assign(rho=np.asarray(rho))).reset_index(drop=True)
    skipped = []
    for c in FEATURES:
        k = int(df[c].isna().sum())
        if k:
            skipped.append(f"dropped {k} rows with missing {c}")
    df = df.dropna(subset=FEATURES)
    if "pitch_count" not in df and {"game_pk", "at_bat_number", "pitch_number"} <= set(df):
        # Fallback only: counted on the filtered frame, so it undercounts. The CLI counts on the raw CSV.
        df = df.assign(pitch_count=pitch_count_in_game(df))
    missing = [c for c in CONTEXT if c not in df]
    if missing:
        skipped.append(f"context: missing columns {missing}")
    train_all, test_all = split_by_game(df)

    rows, ks_rows, clouds, ctx_rows = [], [], {}, []
    for ptype, g in df.groupby("pitch_type"):
        train = train_all[train_all["pitch_type"] == ptype]
        test = test_all[test_all["pitch_type"] == ptype]
        if len(train) < MIN_TRAIN:
            skipped.append(f"{ptype} (n={len(g)}): fewer than {MIN_TRAIN} training pitches")
            continue
        if len(test) == 0:
            skipped.append(f"{ptype} (n={len(g)}): no pitches in held-out games")
            continue
        graded = len(test) >= MIN_TEST
        models = {"gaussian": fit_pitch_type(train, False, seed),
                  "mixture": fit_pitch_type(train, True, seed)}
        for name, m in models.items():
            # Classifier needs a few pitches per class for 5 folds; below 10 report NaN.
            auc = (np.nan,) * 3
            auc8 = np.nan
            if len(test) >= 10:
                draws = [round_like_statcast(sample(m, len(test), seed + k))
                         for k in range(1, N_DRAWS + 1)]
                auc = classifier_auc(test, draws, GRADED, m.axis_center_deg, seed)
                auc8 = classifier_auc(test, draws, FEATURES, m.axis_center_deg, seed)[0]
            rows.append({"pitch_type": ptype, "model": name, "n_train": m.n_train,
                         "n_test": len(test), "n_components": m.gmm.n_components,
                         "auc": auc[0], "auc_lo": auc[1], "auc_hi": auc[2], "auc8": auc8,
                         "graded": graded,
                         "pass": bool(auc[0] <= PASS_AUC) if graded else pd.NA})
            gen_ks = round_like_statcast(sample(m, 1000, seed))  # large draw so the generated side isn't noisy
            ks_rows.append(marginal_ks(test, gen_ks, m.axis_center_deg)
                           .assign(pitch_type=ptype, model=name))

        if not missing:
            ctx_rows.append({"pitch_type": ptype, **context_gain(train, test)})

        # Movement clouds. Calibrate once per type, on the same capped train as M1.
        cal = train.sample(MAX_TRAIN, random_state=seed) if len(train) > MAX_TRAIN else train
        cd, eff = calibrate(cal, "release")
        _, v_s, t_s = fit_plate_crossing(test)
        pos_r, v_rel = release_state(test)
        t_flight = t_s - _time_to_y(test["release_pos_y"].to_numpy(), test["vy0"].to_numpy(),
                                    test["ay"].to_numpy())
        _, vp_r, t_r = simulate_batch(pos_r, v_rel, test["release_spin_rate"].to_numpy(),
                                      test["spin_axis"].to_numpy(), cd, eff,
                                      rho=test["rho"].to_numpy())
        gen = sample(models["mixture"], 1000, seed)
        pos_g, v_g = generated_release_state(gen)
        _, vp_g, t_g = simulate_batch(pos_g, v_g, gen["release_spin_rate"].to_numpy(),
                                      gen["spin_axis"].to_numpy(), cd, eff,
                                      rho=float(train["rho"].median()))
        clouds[ptype] = {"real": test, "generated": gen,
                         "move_statcast": movement_in(v_rel, v_s, t_flight),
                         "move_real_m1": movement_in(v_rel, vp_r, t_r),
                         "move_gen_m1": movement_in(v_g, vp_g, t_g)}

    summary = pd.DataFrame(rows)
    ks = pd.concat(ks_rows, ignore_index=True) if ks_rows else pd.DataFrame()
    context = pd.DataFrame(ctx_rows, columns=["pitch_type", "n_train", "n_test", "n_dropped",
                                              "ll_base", "ll_context"])
    context["ll_gain"] = context["ll_context"] - context["ll_base"]
    return summary, ks, skipped, clouds, context
