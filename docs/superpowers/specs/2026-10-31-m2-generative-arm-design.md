# M2: per-pitcher generative model ("the arm")

Date: 2026-10-31. Project brief: `CLAUDE.md`. M1 spec and results:
`docs/superpowers/specs/2026-10-29-m1-completion-design.md`, `results/M1.md`.

## Intent

The end goal is a virtual Trajekt machine: a gray 3D pitcher on a laptop that throws a
specific MLB pitcher's pitches the way he does. M1 flies a pitch given its starting numbers.
M2 *makes up* starting numbers that look like they came from that pitcher, so M7 can ask
for "a Sale slider, 0-2, his 95th pitch" and get a pitch that can't be told from a real
one at the hand.

Ryan is building this to learn probabilistic ML. Code explains the *why* of every modelling
choice in short comments/docstrings with references. Small steps, Ryan's OK between steps.

What Ryan wants to see when it works: for each pitch type, a cloud of 1,000 real pitches
next to 1,000 generated ones, plus the brief's classifier number.

## What gets modelled

Per pitcher, per pitch type, the joint distribution of what happens **at the hand**:

| Graded feature | Statcast column |
|---|---|
| release side (ft) | `release_pos_x` |
| release height (ft) | `release_pos_z` |
| extension (ft) | `release_extension` |
| velocity (mph) | `release_speed` |
| spin rate (rpm) | `release_spin_rate` |
| spin axis (deg) | `spin_axis` |

- `spin_axis` is **Hawk-Eye measured** (confirmed 2026-10-30 against Savant's
  spin-direction leaderboard; see `CLAUDE.md`). M2 generates the measured axis.
- Spin axis is an angle (0–360). *(Claude's call)* Model it unwrapped around the pitch
  type's circular mean (shifted into mean ± 180°), then wrap samples back to [0, 360).
  Without this, a pitch type whose axis straddles 0/360 gets a nonsense mean.
- Spin efficiency is **not** generated: use M1's fitted per-pitch-type value (re-fitted on
  M2's training games, same 70/30 split, so no held-out leakage). *(Claude's call)* Reuse
  `validation.calibrate(train, "release")` unchanged (Cd and efficiency, same bounds),
  with per-game air density as in M1's `release+game` variant.
- *(Claude's call)* To fly a generated pitch, M1 also needs a launch **direction**. The
  model also includes the vertical and horizontal release angles (from `release_state`
  velocity), as two extra, **ungraded** dimensions. They mostly encode where he was aiming,
  which is M6's job; M6's location model replaces them later. Note this in code.

## Models, in order (brief's order)

1. **Single multivariate Gaussian** per pitch type (mean + full covariance over the 8
   numbers above: 6 graded + 2 angles). *(Claude's call)* Fit with a small diagonal
   regularizer (scikit-learn's `reg_covar`) so thin pitch types stay well-conditioned;
   pitch types with < 30 **training** pitches are skipped and listed.
2. **Gaussian mixture** per pitch type. *(Claude's call)* Number of components 1–4 chosen
   by BIC on the training games, over all 8 dimensions; mixtures only tried when the pitch type has ≥ 100 training
   pitches (fewer → stays a single Gaussian, and the report says so).
3. **Context-conditioned Gaussian** (a later step in this same milestone, built after 1–2
   but **not gated** on them passing; it has its own measure): mean is linear in
   context, covariance shared (i.e. multivariate linear regression with Gaussian
   residuals). Context features:
   - pitch count in the game (fatigue): the pitcher's cumulative pitch number within
     `game_pk`, ordered by `at_bat_number`, `pitch_number`, computed on the **full** CSV
     (all pitch types) before splitting by type;
   - times through the order: `n_thruorder_pitcher`;
   - count: `balls`, `strikes`.
   Measured by mean held-out log-likelihood per pitch vs the unconditioned single Gaussian.
   *(Claude's call)* Rows missing any context value are dropped for this step only, and
   both models are fitted and scored on the same remaining rows (counts reported).
   If context doesn't help for a pitcher/pitch type, the write-up says so; no forcing it.

Sampling: a function that, given a fitted pitcher model, a pitch type, `n`, an optional
context and a seed, returns a DataFrame with the 8 columns (Statcast names for the six).

## Evaluation (all on held-out games)

- **Split:** reuse `pitchsim.validation.split_by_game` (70/30 by game, seed 0), once per
  pitcher before grouping by pitch type, same as M1.
- **Minimum:** pitch types with < 30 pitches (M1's `MIN_PITCHES`) are skipped and listed
  with their count. Pitch types with < 30 **held-out** pitches are fitted and reported but
  marked "too few held-out pitches to grade" (Ohtani has several).
- **Classifier two-sample test (pass/fail):** for each pitch type, held-out real pitches vs
  an equal number of generated pitches, six graded features only. *(Claude's call)*
  Classifier: scikit-learn `HistGradientBoostingClassifier`, 5-fold cross-validated AUC.
  A nonlinear classifier, because a logistic regression can only see mean/linear
  differences, which a fitted Gaussian matches by construction. Spin axis enters as
  sin/cos so the classifier doesn't see a fake jump at 0/360.
  **Rounded like Statcast** (Ryan, 2026-10-31): before any comparison with real pitches
  (AUC and KS), generated values are rounded to Statcast's reporting precision:
  `release_pos_x/z` 0.01 ft, `release_extension` 0.1 ft, `release_speed` 0.1 mph,
  `release_spin_rate` 1 rpm, `spin_axis` 1°. Otherwise the classifier wins on decimals
  alone (Misiorowski FF: 0.997 unrounded vs ~0.66 rounded). The generator itself stays
  unrounded. The derived release angles are continuous on both sides and stay unrounded.
  *(Claude's call)* To steady the number, draw 10 independent generated sets of size
  n_heldout and average the AUC over them.
  **Pass: mean AUC ≤ 0.55 (point estimate)** *(Claude's call)*. Report the AUC with a
  bootstrap 95% interval next to it so small-sample passes are visibly uncertain.
  An 8-feature AUC (graded six + the two angles) is also reported as an ungraded diagnostic.
- **On a fail:** report it honestly in `results/M2.md` with what seems to cause it. The only
  model iteration in scope is the spec's own Gaussian → mixture step; no tuning loops to
  force a pass *(Claude's call)*.
- **Marginals:** per graded feature, two-sample Kolmogorov–Smirnov statistic and p-value,
  held-out real vs generated (spin axis compared after the same unwrapping). Report all;
  flag p < 0.01.
- **Plots (per pitcher, per pitch type):** 1,000 real vs 1,000 generated
  (or all held-out real if fewer): release side vs height, velocity vs spin rate,
  spin axis, extension.
- **Movement (plotted, NOT graded until M3):** three clouds per pitch type:
  (a) real pitches, movement from Statcast's own fit; (b) real held-out pitches flown with
  M1 from their release state; (c) generated pitches flown with M1. The gap (b)→(c) is
  M2's error; (a)→(b) is M1's, which M3 is meant to fix. Plot is labelled
  "not graded until M3". Movement uses one formula for all three, documented in code
  (builder picks, e.g. non-gravity acceleration from a constant-acceleration fit,
  converted to inches of break).
- Each pitcher compares single Gaussian vs mixture (AUC side by side).

## Pitchers and outputs

- Same three as M1, regular season: Ohtani 2025, Sale 2026, Misiorowski 2026, from the
  cached CSVs in `data/raw/`. Nothing hardcoded to these three: fitting takes any pitcher
  CSV.
- Library code in `src/pitchsim/generative.py` (brief's layout). CLI in `scripts/`
  (brief: "train"). Outputs per pitcher in `results/m2/<slug>/` (summary CSV + plots);
  write-up `results/M2.md`: what was measured, numbers per pitcher × pitch type × model,
  context step's log-likelihood gain, what's still wrong. Every number reproducible from
  `scripts/`.
- Same synthetic guard as M1: synthetic data only ever writes to `results/synthetic/`.
- Tests (synthetic data, labelled): sampling reproduces a known Gaussian's mean/covariance;
  spin-axis wrapping round-trips across 0/360; the two-sample test gives AUC ≈ 0.5 when
  "generated" is a second draw from the same distribution and clearly > 0.5 when it's
  shifted; pitch count in game is computed per game in pitch order.
- Small fixes riding along: `statcast.py` module docstring and README limitation #2 still
  say measured-vs-inferred `spin_axis` is unresolved; update them to "measured".

## Constraints

- Only new dependency: `scikit-learn` (brief schedules it for M2).
- Brief's conventions: type hints, docstrings, units, fixed seeds, `results/*.md` committed,
  `data/` and other results gitignored.
- Commits carry **no `Co-Authored-By` trailer** (Ryan's standing rule).

## Non-goals

- No location/aim model (M6), no seam-shifted-wake force (M3), no viewer/export (M4).
- No pooling across pitchers (hierarchical model). It's the upgrade for scaling to many
  pitchers, where rare pitch types have few samples; noted in `results/M2.md` as future
  work, not built.
- No saving fitted models to disk; fitting takes seconds. Add when M7 needs it.
- No splitting a season by date for mid-season arsenal changes.

## Anti-goals

- Generated pitches presented as real, or synthetic results mixed with real ones.
- Any fitting, BIC selection or classifier training that sees held-out games' real pitches,
  except the classifier test itself (whose only real input is held-out pitches).
- Grading M2 on movement or plate location, which would blame M2 for M1/M3/M6 errors.

## Left to the builder

- Function names and signatures inside `generative.py`; whether the evaluation lives there
  or in a sibling module.
- The movement formula (one formula for all three clouds, documented).
- Plot layout, following M1's report style.
- How the CLI loops over pitchers.
