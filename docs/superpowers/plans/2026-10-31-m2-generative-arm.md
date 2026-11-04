# M2 Generative Arm Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fit a per-pitcher, per-pitch-type generative model of how each pitch leaves the hand (Gaussian, then mixture, then context-conditioned), grade it on held-out games with a classifier two-sample test, and write up `results/M2.md` for Ohtani 2025, Sale 2026, Misiorowski 2026.

**Architecture:** `src/pitchsim/generative.py` holds features, fitting, sampling, the two-sample tests, the per-pitcher evaluation and the context model. `scripts/train_generative.py` is a thin CLI that runs it per pitcher and draws plots. Single Gaussian and mixture share one code path (`GaussianMixture` with `n_components=1` is a single Gaussian).

**Tech Stack:** Python 3.11+, numpy, pandas, scipy, matplotlib, scikit-learn (new), existing `pitchsim` modules.

**Spec:** `docs/superpowers/specs/2026-10-31-m2-generative-arm-design.md`

## Global Constraints

- Graded features, in this order: `release_pos_x, release_pos_z, release_extension, release_speed, release_spin_rate, spin_axis`. Ungraded extra dims: `release_angle_v_deg, release_angle_h_deg`.
- Spin axis is modelled unwrapped around the pitch type's training circular mean (into mean ± 180°) and wrapped back to [0, 360) on sampling.
- Split: `pitchsim.validation.split_by_game(df)` (70/30, seed 0), once per pitcher, before grouping by pitch type.
- Skip pitch types with < 30 **training** pitches (listed with counts). Fewer than 30 held-out pitches → fitted, reported, `graded = False`, `pass` empty.
- Mixture: components 1–4 by BIC on all 8 dims, only when ≥ 100 training pitches.
- Pass: mean AUC over 10 generated draws ≤ 0.55 (six graded features, spin axis as sin/cos, `HistGradientBoostingClassifier`, 5-fold CV). Bootstrap 95% interval reported. 8-feature AUC reported as an ungraded diagnostic.
- No tuning loops to force a pass; fails are reported.
- Fixed seeds (`seed=0`). No model, BIC or calibration ever sees held-out real pitches.
- Only new dependency: `scikit-learn`. Every modelling choice gets a short *why* comment (Ryan is learning from this code).
- Synthetic data writes only to `results/synthetic/`. Outputs per real pitcher in `results/m2/<slug>/`.
- Commits: plain messages, **no `Co-Authored-By` trailer**, ever.
- Stop for Ryan's OK between tasks.

## Review Focus

1. Pitch type whose spin axis straddles 0/360 → mean lands near 0/360, samples wrap correctly (test in Task 1).
2. Pitch type with 30–99 training pitches → single Gaussian only, `n_components = 1`; < 30 training → listed in skipped, not silently dropped (test in Task 3).
3. Pitch type with < 30 held-out pitches → row present with `graded = False` and empty `pass`, never `pass = True` (test in Task 3).
4. NaN in a feature column (`release_extension` isn't in `statcast.REQUIRED`) or a context column → rows dropped for that step only, counts reported (tests in Tasks 3 and 5).
5. Synthetic CSV passed to the CLI → refuses any output folder outside `results/synthetic/` (check in Task 4).

---

### Task 1: Features, fit and sample (`generative.py` core)

**Files:**
- Modify: `pyproject.toml` (add `scikit-learn` to dependencies)
- Create: `src/pitchsim/generative.py`
- Test: `tests/test_generative.py`

**Interfaces:**
- Consumes: `statcast.release_state(df)`.
- Produces:
  - `GRADED: list[str]`, `ANGLES: list[str]`, `FEATURES = GRADED + ANGLES`
  - `MIN_TRAIN = 30`, `MIN_TEST = 30`, `MIN_MIXTURE = 100`, `MAX_COMPONENTS = 4`
  - `add_release_angles(df) -> pd.DataFrame` — copy with `release_angle_v_deg = deg(atan2(vz, -vy))`, `release_angle_h_deg = deg(atan2(vx, -vy))` from `release_state` velocity.
  - `circular_mean_deg(a: np.ndarray) -> float`; `unwrap_deg(a, center) -> np.ndarray` (into center ± 180).
  - `@dataclass PitchTypeModel: gmm: GaussianMixture; axis_center_deg: float; n_train: int`
  - `fit_pitch_type(train: pd.DataFrame, mixture: bool, seed: int = 0) -> PitchTypeModel` — `covariance_type="full"`, default `reg_covar`; if `mixture` and `len(train) >= MIN_MIXTURE`, pick 1..`MAX_COMPONENTS` by lowest BIC, else 1 component.
  - `sample(model: PitchTypeModel, n: int, seed: int) -> pd.DataFrame` — columns `FEATURES`, `spin_axis` wrapped to [0, 360).

- [ ] **Step 1: Failing tests** (synthetic data, say so in each docstring)

```python
def test_sample_matches_fitted_gaussian():
    """Synthetic: draws from a known 8-D Gaussian; refit+sample recovers mean and cov."""
    rng = np.random.default_rng(0)
    mean = np.array([-1.9, 5.9, 6.3, 95, 2400, 210, -1.5, 1.0])
    cov = np.diag([0.01, 0.01, 0.02, 0.6, 3600, 36, 0.3, 0.3])
    train = pd.DataFrame(rng.multivariate_normal(mean, cov, 5000), columns=FEATURES)
    out = sample(fit_pitch_type(train, mixture=False), 20000, seed=1)
    assert np.allclose(out[FEATURES].mean(), mean, rtol=0.01, atol=0.05)
    assert np.allclose(out[FEATURES].std(), np.sqrt(np.diag(cov)), rtol=0.05)

def test_spin_axis_wraps_across_zero():
    """Synthetic: axis centered on 355 deg, spread 10 deg."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(0, 1, (3000, 8)), columns=FEATURES)
    df["spin_axis"] = (355 + rng.normal(0, 10, 3000)) % 360
    m = fit_pitch_type(df, mixture=False)
    assert abs(((m.axis_center_deg - 355) + 180) % 360 - 180) < 1
    out = sample(m, 5000, seed=1)["spin_axis"]
    assert out.between(0, 360, inclusive="left").all()
    assert abs(((circular_mean_deg(out.to_numpy()) - 355) + 180) % 360 - 180) < 1

def test_mixture_finds_two_clusters_and_respects_minimum():
    """Synthetic: two well-separated velocity clusters."""
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, (400, 8)); b = rng.normal(0, 1, (400, 8)); b[:, 3] += 8
    df = pd.DataFrame(np.vstack([a, b]), columns=FEATURES); df["spin_axis"] += 200
    assert fit_pitch_type(df, mixture=True).gmm.n_components == 2
    assert fit_pitch_type(df.iloc[:60], mixture=True).gmm.n_components == 1

def test_release_angles_from_synthetic():
    df = add_release_angles(make(n=200, seed=0))
    assert df["release_angle_v_deg"].between(-10, 5).all()
```

- [ ] **Step 2:** `pytest tests/test_generative.py -v` → FAIL (module missing).
- [ ] **Step 3:** Add `scikit-learn` to `pyproject.toml`; `pip install -e .[dev]`. Implement the Interfaces above. Comments explain: why unwrap (a mean of 359° and 1° is 180° otherwise), why BIC (penalises extra components so it doesn't fit noise), why full covariance (release height and extension move together), why the angles are ungraded (aim is M6).
- [ ] **Step 4:** `pytest -q` → all pass.
- [ ] **Step 5:** Commit `"generative.py: per-pitch-type Gaussian / mixture fit and sampling"`.

---

### Task 2: Two-sample tests

**Files:**
- Modify: `src/pitchsim/generative.py`
- Test: `tests/test_generative.py`

**Interfaces:**
- Produces:
  - `N_DRAWS = 10`, `PASS_AUC = 0.55`
  - `classifier_auc(real: pd.DataFrame, generated: list[pd.DataFrame], cols: list[str], center: float, seed: int = 0) -> tuple[float, float, float]` — for each generated set (same size as `real`), `HistGradientBoostingClassifier(random_state=seed)` with `cross_val_predict(..., cv=StratifiedKFold(5, shuffle=True, random_state=seed), method="predict_proba")`; AUC of out-of-fold probabilities; 200 bootstrap resamples of those predictions for a 2.5/97.5 interval. Returns `(mean AUC, mean lo, mean hi)` across sets. `spin_axis` in `cols` becomes `sin`, `cos` columns.
  - `marginal_ks(real: pd.DataFrame, generated: pd.DataFrame, center: float) -> pd.DataFrame` — rows `feature, ks_stat, p_value, flagged (p < 0.01)` for each of `GRADED`, `spin_axis` compared after `unwrap_deg(..., center)`.

- [ ] **Step 1: Failing tests**

```python
def _gauss(n, shift=0.0, seed=0):
    """Synthetic 8-D sample; shift moves release_speed by `shift` mph."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(rng.normal(0, 1, (n, 8)), columns=FEATURES)
    df["release_speed"] += 95 + shift; df["spin_axis"] = (200 + 5 * df["spin_axis"]) % 360
    return df

def test_auc_near_half_for_same_distribution():
    real = _gauss(300)
    auc, lo, hi = classifier_auc(real, [_gauss(300, seed=s) for s in range(1, 4)], GRADED, 200)
    assert auc < 0.6 and lo < 0.5 < hi

def test_auc_high_for_shifted_distribution():
    auc, _, _ = classifier_auc(_gauss(300), [_gauss(300, shift=2, seed=1)], GRADED, 200)
    assert auc > 0.8

def test_marginal_ks_flags_shifted_feature_only():
    ks = marginal_ks(_gauss(500), _gauss(500, shift=2, seed=1), 200).set_index("feature")
    assert ks.loc["release_speed", "flagged"] and not ks.loc["release_spin_rate", "flagged"]
```

- [ ] **Step 2:** `pytest tests/test_generative.py -v` → the three new tests FAIL.
- [ ] **Step 3:** Implement. Comments: why a nonlinear classifier (a fitted Gaussian matches means and linear correlations by construction, so logistic regression would pass it trivially); why sin/cos (a fake jump at 0/360 is a free tell); why average several draws (one draw of a few dozen pitches is noisy); reference: Lopez-Paz & Oquab, "Revisiting Classifier Two-Sample Tests" (ICLR 2017).
- [ ] **Step 4:** `pytest -q` → all pass.
- [ ] **Step 5:** Commit `"generative.py: classifier two-sample test and marginal KS"`.

---

### Task 3: Per-pitcher evaluation and movement

**Files:**
- Modify: `src/pitchsim/generative.py`
- Test: `tests/test_generative.py`

**Interfaces:**
- Consumes: `split_by_game`, `calibrate(train, "release")` (needs a `rho` column), `simulate_batch`, `release_state`, `fit_plate_crossing`, `physics.G`, `validation.FT_PER_S_TO_MPH`.
- Produces:
  - `movement_in(vel_release: np.ndarray, vel_plate: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]` — horizontal and vertical break (in): average non-gravity acceleration over the flight, `a = (v_plate - v_release)/t` with `G` added back to z, times `0.5 * t**2 * 12`. One formula for all three clouds.
  - `generated_release_state(gen: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]` — position `(release_pos_x, 60.5 - release_extension, release_pos_z)`; velocity of magnitude `release_speed / FT_PER_S_TO_MPH` along the two angles.
  - `evaluate_pitcher(df: pd.DataFrame, rho: np.ndarray, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict]` — returns `(summary, ks, skipped, clouds)`.
    - `summary`: one row per pitch type × model (`"gaussian"`, `"mixture"`), columns `pitch_type, model, n_train, n_test, n_components, auc, auc_lo, auc_hi, auc8, graded, pass`.
    - `ks`: `marginal_ks` rows plus `pitch_type, model`.
    - `clouds[pitch_type]`: dict with `real` (held-out rows), `generated` (1,000 from the mixture model), `move_statcast`, `move_real_m1`, `move_gen_m1` (each `(hb, vb)` arrays).
  - Steps inside: drop rows with NaN in `FEATURES` (count into `skipped` text), `add_release_angles`, split once, per type: skip if `< MIN_TRAIN` train; fit both models on train; `graded = n_test >= MIN_TEST`; AUC on `GRADED` and on `FEATURES`; `pass = auc <= PASS_AUC` if graded else `pd.NA`. Movement: `cd, eff = calibrate(train.assign(rho=...), "release")` (`MAX_TRAIN` cap as in M1); real held-out flown with their own rho; generated flown with the median training rho.

- [ ] **Step 1: Failing tests** (synthetic `make`, labelled)

```python
def test_evaluate_pitcher_on_synthetic():
    """Synthetic pitcher, real-data path. Model is exact-ish, so most types pass."""
    df = make(n=1500, seed=0)
    summary, ks, skipped, clouds = evaluate_pitcher(df, np.full(len(df), air_density()))
    g = summary[summary.model == "gaussian"]
    assert (g.auc < 0.65).all() and set(g.pitch_type) == {"FF", "SL", "CH", "CU"}

def test_thin_types_skipped_or_ungraded():
    df = make(n=1500, seed=0)
    cu = df.index[df.pitch_type == "CU"]
    df = df.drop(cu[20:])                            # 20 CU total -> < 30 train
    ch = df.index[df.pitch_type == "CH"]
    df = df.drop(ch[70:])                            # ~70 CH -> < 30 held-out
    summary, _, skipped, _ = evaluate_pitcher(df, np.full(len(df), air_density()))
    assert any(s.startswith("CU") for s in skipped)
    ch_rows = summary[summary.pitch_type == "CH"]
    assert (~ch_rows.graded).all() and ch_rows["pass"].isna().all()
    assert (summary[summary.pitch_type == "CH"].n_components == 1).all()

def test_nan_extension_rows_dropped():
    df = make(n=1500, seed=0); df.loc[:9, "release_extension"] = np.nan
    _, _, skipped, _ = evaluate_pitcher(df, np.full(len(df), air_density()))
    assert any("10" in s and "release_extension" in s for s in skipped)

def test_movement_formula_matches_statcast_fit():
    """For a constant-acceleration path the formula returns 0.5*a*t^2 exactly."""
    df = make(n=50, seed=0)
    _, v_rel = release_state(df); _, v_plate, t_plate = fit_plate_crossing(df)
    t_rel = ...  # flight time = t_plate - t at release (from statcast._time_to_y)
    hb, vb = movement_in(v_rel, v_plate, t_plate - t_rel)
    assert np.allclose(hb, 0.5 * df.ax * (t_plate - t_rel) ** 2 * 12)
```

(Fill `t_rel` with `statcast._time_to_y(df.release_pos_y, df.vy0, df.ay)`.)

- [ ] **Step 2:** `pytest tests/test_generative.py -v` → new tests FAIL.
- [ ] **Step 3:** Implement. `skipped` strings follow M1's form: `"<type> (n=<count>): <reason>"`; the NaN drop reads `"dropped <k> rows with missing <column>"`. Docstring on `evaluate_pitcher` states movement is not graded (M1/M3's job) and why `(b)→(c)` isolates M2's error.
- [ ] **Step 4:** `pytest -q` → all pass. If synthetic AUCs exceed 0.65, report the numbers rather than loosening the bound.
- [ ] **Step 5:** Commit `"generative.py: per-pitcher evaluation and movement clouds"`.

---

### Task 4: CLI and plots

**Files:**
- Create: `scripts/train_generative.py`
- Test: run on synthetic

**Interfaces:**
- Consumes: `evaluate_pitcher`, `data.fetch_game_conditions`, `data.add_air_density`, `statcast.clean`, `physics.air_density`.
- Produces: `python scripts/train_generative.py <csv> [--out DIR] [--no-weather]`. Default `--out results/m2/<player_name slug>`; synthetic (`"Synthetic" in player_name`) refuses anything outside `results/synthetic/`. Writes `summary.csv`, `ks.csv`, `skipped.txt`, and per pitch type `<type>.png`: release side vs height, velocity vs spin rate, spin axis histogram, extension histogram (real held-out vs generated), plus a movement panel with the three clouds titled "movement — not graded until M3". Prints the summary table.

- [ ] **Step 1:** Implement, reusing `scripts/validate.py`'s slug, synthetic guard and plot `style` (import or copy the few lines; don't refactor validate.py).
- [ ] **Step 2:** `python -m pitchsim.synthetic; python scripts/train_generative.py data/synthetic.csv --no-weather` → writes `results/synthetic/` files; `python scripts/train_generative.py data/synthetic.csv --no-weather --out results/m2/x` → exits with the refusal message.
- [ ] **Step 3:** `pytest -q` → all pass.
- [ ] **Step 4:** Commit `"train_generative.py: CLI and per-pitch-type plots"`.

---

### Task 5: Context-conditioned model

**Files:**
- Modify: `src/pitchsim/generative.py`, `scripts/train_generative.py` (write `context.csv`)
- Test: `tests/test_generative.py`

**Interfaces:**
- Produces:
  - `CONTEXT = ["pitch_count", "n_thruorder_pitcher", "balls", "strikes"]`
  - `pitch_count_in_game(df: pd.DataFrame) -> pd.Series` — 1-based count within `game_pk`, ordered by `at_bat_number, pitch_number`, aligned to `df.index`. Computed on the full CSV before any filtering.
  - `@dataclass ContextModel: coef: np.ndarray; cov: np.ndarray; axis_center_deg: float` — mean = `[1, context] @ coef` over the six `GRADED` (spin axis unwrapped), shared residual covariance.
  - `fit_context(train: pd.DataFrame) -> ContextModel` (least squares, `np.linalg.lstsq`).
  - `sample_context(model: ContextModel, context: pd.DataFrame, seed: int) -> pd.DataFrame` — one draw per context row.
  - `context_gain(train, test) -> dict` — keys `n_train, n_test, n_dropped, ll_base, ll_context` (mean held-out log-likelihood per pitch, `scipy.stats.multivariate_normal`). Rows missing any `CONTEXT` value dropped from both; baseline is the intercept-only fit on the same rows.
  - `evaluate_pitcher` gains a fifth return value `context: pd.DataFrame` (one row per pitch type).
- Ruling carried from the spec: the context step scores the six graded features only; the angles encode aim, which depends on count, and that's M6's job.

- [ ] **Step 1: Failing tests**

```python
def test_pitch_count_in_game_orders_within_game():
    df = pd.DataFrame({"game_pk": [2, 1, 1, 2, 1], "at_bat_number": [1, 2, 1, 1, 1],
                       "pitch_number": [1, 1, 2, 2, 1]})
    assert pitch_count_in_game(df).tolist() == [1, 3, 2, 2, 1]

def _fatigue_frame(n=2000):
    """Synthetic: velocity drops 0.03 mph per pitch thrown."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(0, 1, (n, 6)), columns=GRADED)
    df["spin_axis"] = (200 + 5 * df["spin_axis"]) % 360
    df["pitch_count"] = rng.integers(1, 110, n); df["n_thruorder_pitcher"] = 1 + df.pitch_count // 35
    df["balls"] = rng.integers(0, 4, n); df["strikes"] = rng.integers(0, 3, n)
    df["release_speed"] += 97 - 0.03 * df["pitch_count"]
    return df

def test_context_gain_detects_fatigue():
    df = _fatigue_frame()
    r = context_gain(df.iloc[:1400], df.iloc[1400:])
    assert r["ll_context"] > r["ll_base"]

def test_context_gain_drops_missing_rows():
    df = _fatigue_frame().astype({"n_thruorder_pitcher": float})
    df.loc[1400:1404, "n_thruorder_pitcher"] = np.nan
    r = context_gain(df.iloc[:1400], df.iloc[1400:])
    assert r["n_dropped"] == 5 and r["n_test"] == 595
```

- [ ] **Step 2:** `pytest tests/test_generative.py -v` → new tests FAIL.
- [ ] **Step 3:** Implement. Comments: why linear-in-context with shared covariance (the simplest model where context shifts the cloud; it's multivariate linear regression with Gaussian residuals), why log-likelihood is the right measure (it scores the whole distribution, not just the mean). CLI writes `context.csv`.
- [ ] **Step 4:** `pytest -q` → all pass; synthetic CLI run still works.
- [ ] **Step 5:** Commit `"generative.py: context-conditioned Gaussian (fatigue, times through order, count)"`.

---

### Task 6: Real pitchers and `results/M2.md`

**Files:**
- Create: `results/M2.md`
- Modify: `README.md` (M2 commands + results section; limitation #2 → spin_axis measured), `src/pitchsim/statcast.py` (module docstring: spin_axis measured, per CLAUDE.md)

- [ ] **Step 1:** Run on the cached CSVs (no refetch):
  `python scripts/train_generative.py data/raw/660271_2025-03-01_2025-11-30.csv`;
  `... data/raw/519242_2026-03-01_2026-11-30.csv`; `... data/raw/694819_2026-03-01_2026-11-30.csv`.
  Expected: `results/m2/<slug>/` per pitcher.
- [ ] **Step 2:** Write `results/M2.md`: what was measured (held-out games, 70/30, seed 0, six graded features, pass = mean AUC ≤ 0.55 over 10 draws); per pitcher × pitch type table of n_train, n_test, components, Gaussian AUC [interval], mixture AUC [interval], pass, flagged KS features; 8-feature diagnostic AUC; context table (ll_base vs ll_context, n_dropped); skipped and ungraded types; movement gap observations labelled ungraded; what's still wrong; future work: pooling across pitchers for scale. Every number copied from the CSVs.
- [ ] **Step 3:** README and `statcast.py` docstring updates.
- [ ] **Step 4:** `pytest -q` → all pass.
- [ ] **Step 5:** Commit `"M2 results: Ohtani, Sale, Misiorowski"`.
