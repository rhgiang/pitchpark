# M3: seam-shifted wake (the push spin can't explain)

Date: 2026-11-04. Project brief: `CLAUDE.md`. Builds on M1 (`results/M1.md`) and M2
(`results/M2.md`, `docs/superpowers/specs/2026-10-31-m2-generative-arm-design.md`).

## Intent

M1 flies a pitch with gravity, drag and Magnus (spin) and lands four-seamers 2–4 in from the
real crossing, but some pitches miss far more, each in one consistent direction: Ohtani's
sweeper 9.1 in (7.6 in too low), sinker 8.9 in, splitter 7.6 in; Sale's sinker 5.0 in
(4.3 in too high). These are the pitch types where the seams bend the air flow and push the
ball somewhere its spin axis doesn't predict (seam-shifted wake; Barton Smith / Driveline).
M3 adds that missing push so these pitches fly like the real ones, for real pitches and for
M2's generated ones, so M7's made-up pitches benefit too.

Ryan is building this to learn. Code explains the physics and modelling *why* in short
comments with references. Small steps, Ryan's OK between steps.

## Goals

- **Targets (Ryan):** realistic bar: each seam-affected pitch type gets about as accurate
  as the same pitcher's four-seam fastball (median held-out plate miss). Stretch: 1–2 in.
  Report honestly against both.
- **Per-pitch, not fixed (Ryan):** the extra push varies pitch to pitch with how the ball
  is released, and the model says so. A fixed-push-per-pitch-type baseline is fitted too;
  if the per-pitch model doesn't beat it on held-out games, the write-up says the data
  shows a fixed habit.

## The model

- **What's predicted:** per pitch, the extra constant acceleration (x and z, ft/s²) that
  would close M1's calibrated plate miss: a = 2·miss / t², with t the simulated flight
  time. *(Claude's call)* The target is derived from the same plate miss M3 is graded on,
  so fitting and grading can't disagree. It is honestly "what's left after M1's
  calibration": M1's Cd and efficiency already soak up part of the seam push (4 of 15
  pitch types sit on the 0.20 efficiency floor), and a joint refit is a non-goal.
  Builder checks numerically that adding the fitted constant push to one pitch closes its
  miss to within a small tolerance (drag makes a = 2·miss/t² approximate) and documents it.
- **Inputs, known at the hand only:** measured `spin_axis` (as sin/cos), `release_spin_rate`,
  `release_speed`, `release_pos_x`, `release_pos_z`, `release_extension`. These are M2's
  generated features, so the model works on made-up pitches. **A pitch's own movement or
  inferred axis is never an input**: that would predict the movement from the movement.
- **Form:** linear, per pitcher × pitch type, fitted on training games (CLAUDE.md: "start
  linear"). Something nonlinear only if linear clearly falls short, and as a later,
  separate step. *(Claude's call)* Ridge, with its strength chosen by cross-validation
  **by game inside the training games**, so held-out games are never touched. The push
  is in world x/z; within one pitch type the axis varies by tens of degrees at most, so
  sin/cos inputs are enough to let it turn with the axis, no rotated frame.
- **Fixed-push baseline:** one constant (x, z) vector per pitcher × pitch type, the
  training-game mean target. The per-pitch model is that plus slopes, so the comparison
  is nested: the slopes have to earn their place on held-out games.
- **Where it acts:** a constant extra acceleration added to M1's equations of motion
  (`physics.py`), so trajectories stay physical for the M4 viewer. Not a post-hoc shift of
  the plate crossing. *(Claude's call)*
- **Order of fitting:** M1's Cd and spin efficiency are calibrated first on the training
  games (unchanged), then M3 is fitted on what's left, on the same training games.
  *(Claude's call)*
- **Applies to every pitch type**, fastballs included. No hand-picked list of "affected"
  pitches; the numbers show where it matters.

## Evaluation (held-out games only)

- **Split:** `split_by_game`, 70/30, seed 0, once per pitcher, same as M1/M2.
- **Main number:** per pitcher × pitch type, median held-out plate miss (in) and bias x/z
  for three variants: **M1 only**, **M1 + fixed push**, **M1 + per-pitch push**. Same
  target as M1 (Statcast fit crossing at y = 17/12 ft), release start, per-game air density.
- **Is the drop real:** a paired bootstrap interval on the per-pitch improvement
  (M1 miss − M3 miss on the same held-out pitches). *(Claude's call)*
- **Pass rule (Ryan, 2026-11-04):** a graded pitch type passes if the median held-out
  miss drops by at least 0.6 in and the 95% paired bootstrap interval on the improvement
  is above zero (CLAUDE.md: "shrinks measurably"). The 0.6 in floor is the largest drop
  seen on a synthetic pitch type with no seam force, where the push only removes a
  constant M1 offset. The fastball-level and 1–2 in targets are reported
  as goals beside it; they don't decide the pass.
- **Grading minimum:** pitch types with < 30 held-out pitches are fitted and reported but
  marked too few to grade (M1/M2's rule), and so are types with one held-out game, where
  the interval collapses to a point. Held-out game counts are reported. Ohtani's sweeper (27), splitter (7), sinker (5)
  fall under this.
- **Per-pitch vs fixed:** per pitch type, is the per-pitch model's held-out miss lower
  than the fixed push's, with the same paired bootstrap.
- **Link to the axis gap (reported, not graded):** per pitch type, the training-game mean
  gap between measured `spin_axis` and the movement-inferred axis, next to the size and
  direction of the learned push. The brief frames M3 as "movement as a function of that
  gap"; this shows whether the learned push lines up with it.
- **M2 knock-on (reported, not graded):** fly M2-generated pitches with M1 + M3 and
  compare their movement with held-out Statcast movement (KS per component, same
  `movement_in` formula). M2's movement plots were "not graded until M3"; this is the
  first look. *(Claude's call: report, not pass/fail; grading generated movement is a new
  rule for Ryan to set once the numbers are seen.)*

## Pitchers and data

- Existing three: Ohtani 2025, Sale 2026, Misiorowski 2026 (cached).
- **Added (Ryan):** Cristopher Sánchez 2026 and Michael King 2026, regular season, both
  sinker/changeup-heavy (King also throws a sweeper), so the seam-affected pitch types have
  enough held-out pitches to grade. Fetched once with `scripts/fetch.py`, cached in
  `data/raw/`. *(Claude's call)* After fetching, print held-out counts per pitch type and
  continue; if either has a thin 2026 season, the write-up says so; no swapping pitchers
  without Ryan.
- Nothing hardcoded to these five.

## Outputs

- Library code in `src/pitchsim/seam.py` (brief's layout); physics hook in `physics.py`.
- CLI in `scripts/` (builder names it). Per pitcher in `results/m3/<slug>/`: summary CSV
  (three variants side by side), plots. Write-up `results/M3.md`: what was measured,
  numbers, the realistic and stretch targets, per-pitch vs fixed verdict, what's still
  wrong.
- M1 results for the two new pitchers come for free from the "M1 only" variant; no
  separate M1 rerun or `results/M1.md` edit. *(Claude's call)*
- Tests (synthetic, labelled): with no extra push, the physics output is unchanged; a
  known constant push added to synthetic pitches is recovered by the fit; the per-pitch
  model recovers a known linear dependence on spin axis; the inputs used for fitting never
  include movement columns (`ax`, `az`, `pfx_*`, `plate_*`).
- Synthetic data only in `results/synthetic/`.

## Ride-along cleanups (from M2's ledger; their own first step and commit, *Claude's call*)

- Speed up the M2 test file (shared fitted fixture instead of refitting per test); M3 adds
  more tests and the suite already takes ~9 min.
- `auc_margin` shows 0.020 for pitch types with no AUC; make it blank (NaN). No M2 rerun
  is needed beyond what the M2 script does on its next run.

## Constraints

- No new dependencies (numpy, scipy, pandas, scikit-learn already present).
- Brief conventions: type hints, docstrings, units, fixed seeds, every number reproducible
  from `scripts/`, `results/*.md` committed, `data/` and other results gitignored.
- Commits carry **no `Co-Authored-By` trailer** (Ryan's standing rule).

## Non-goals

- Joint refit of Cd / spin efficiency with the M3 push.
- A nonlinear model (later step only if linear clearly falls short).
- Pooling across pitchers.
- Seam-orientation physics from first principles (ball orientation isn't in Statcast).
- Location/aim (M6), viewer (M4).

## Anti-goals

- Using a pitch's own movement (ax/az, pfx, plate location, inferred axis) as an input.
- Any fitting that sees held-out games.
- Hand-picking which pitch types get the push.
- Calling a pitch type fixed when it has < 30 held-out pitches.

## Left to the builder

- How the per-pitch extra acceleration target is computed (documented, physical units).
- Regression details (plain least squares vs light ridge) and function names.
- Plot layout, following M1/M2 report style; CLI name and pitcher looping.
