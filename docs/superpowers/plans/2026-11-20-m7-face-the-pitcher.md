# M7: Face the Pitcher Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** a "Face him" mode in the viewer where any of the 8 pitchers throws newly generated, sampled
at-bats (M6 picks → M2 generates → M1/M3 fly → M5 throws → M4 shows), with the checks that say how
real they are.

**Architecture:** Python fits every model on training games and writes one static JSON per
pitcher (`web/public/data/face/<slug>.json`):
- model parameters: pitch-type logistic, location Gaussians, hitter model, zone;
- a bank of pre-flown M2 + M1/M3 pitches per type.

A pure JS engine (`web/src/face/engine.js`) samples type → target → bank pitch → re-aim → result →
count. The viewer plays its pitches through the existing state/HUD as "generated" pitches. At-bat
realism is checked by running that same engine in Node against real held-out at-bat statistics
exported by Python.

**Tech Stack:** Python 3.11 (numpy, pandas, scipy, scikit-learn: already installed), Node test
runner, Vite + Three.js. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-11-20-m7-face-the-pitcher-design.md`. Read it first: its
"Rulings from the builder check" section is binding.

## Global Constraints

- **Worktree:** `C:\Users\ryanh\Documents\arkitecht-m7`, branch `m7-face-the-pitcher`.
- **Python:** `PYTHONPATH=src ../arkitecht/.venv/Scripts/python.exe` (the editable install points
  at the main checkout).
- **Raw data:** `data/raw/` is a directory junction to `../arkitecht/data/raw` (Task 1). Never
  re-download cached data.
- **Pitchers (slug → MLBAM id):** sale-chris 519242, ohtani-shohei 660271,
  misiorowski-jacob 694819, cease-dylan 656302, yamamoto-yoshinobu 808967, king-michael 650633,
  sanchez-cristopher 650911, schlittler-cam 693645. M2 and M3 are per season (newest season);
  M6 pools seasons, as the existing scripts already do.
- **Splits:**
  - Every model is fitted on `validation.split_by_game(seed=0)` **training** games.
  - Every M7 check uses the held-out games.
  - Fixed seeds; seed 0 unless stated.
- **Real vs simulated:**
  - Every generated pitch carries `generated: true`, and the UI labels it "Simulated".
  - Synthetic-pitcher exports may only be written under `web/test/fixtures/` (same rule as
    `scripts/export_viewer.py`).
- **Sampling:** type and location are always sampled, never argmax.
- **Fixed models:**
  - **Sequencing:** logistic + times through the order, features `["count", "stand", "prev_type",
    "tto"]`. Face him uses `tto = "1"`.
  - **Arm:** the M2 mixture model (`fit_pitch_type(train, mixture=True)`). Context conditioning
    is not used.
- **Size budget:** each `face/<slug>.json` ≤ 1 MB gzipped (test enforces).
- **Frozen code:** do not modify `web/src/signed_off/` (Ohtani) or `scripts/blender/`.
- **Replay mode:** unchanged in behaviour and look. The existing web tests keep passing untouched.
- **Commits:** `git add <paths>` (never `-A`). **No Co-Authored-By or any co-author trailer.**
- **Docstrings:** units in names or docstrings; a short "why" comment with a reference for each
  modelling choice (CLAUDE.md).

## Review Focus

1. **Two-strike fouls.** A foul (or foul bunt) with 2 strikes leaves the count unchanged; a foul tip
   with 2 strikes is a strikeout. → Task 6 test `two-strike foul keeps count, foul tip strikes out`.
2. **A pitch type with too few pitches** (e.g. a 1% splitter): it is left out and the probabilities
   are renormalised. It must never be sampled with an empty bank. → Task 5 test
   `rare type absent from bank is never in seq classes`; Task 6 test
   `typeProbs renormalises over bank types`.
3. **Switching pitcher, batter side or mode in the middle of an at-bat.** This starts a fresh at-bat
   with no stale count, and Replay state is restored exactly when you switch back. → Task 9 test
   `switching back to replay restores ab and pitch`.
4. **A re-aimed target far outside the zone** (location sample in the 3-sigma tail, e.g. 3 ft wide).
   The flight must stay finite, end at the plate, and give a "ball". → Task 6 test
   `far target re-aims to the exact plate point`.
5. **Guessing on with no input before the pitch lands.** It counts as no call (not wrong), and play
   never stalls. → Task 10 test `no call scores as skipped`.

---

### Task 1: Worktree data link + M2 for the five new pitchers

**Files:**
- Modify: `results/M2.md` (gitignored folder: commit only if already tracked; check with
  `git ls-files results`)
- Create: `data/raw` junction (not committed)

- [ ] **Step 1:** In PowerShell, create the data link:
  `cmd /c mklink /J C:\Users\ryanh\Documents\arkitecht-m7\data\raw C:\Users\ryanh\Documents\arkitecht\data\raw`.
  Check that `ls data/raw` lists the 8 pitchers' CSVs.
- [ ] **Step 2:** Run M2 for each new pitcher on his newest season:
  `PYTHONPATH=src ../arkitecht/.venv/Scripts/python.exe scripts/train_generative.py data/raw/<id>_2026-03-01_2026-11-30.csv`.
  - Do this for 656302, 808967, 650633, 650911 and 693645.
  - Rerun even where `results/m2/<slug>` exists, so all eight use today's code.
  - Expected output: `results/m2/<slug>/summary.csv` for each.
- [ ] **Step 3:** Add the five pitchers' graded rows to the results table in `results/M2.md`, in its
  existing format. Add one paragraph stating, per pitch type, which pass and which fail under the
  floor-relative rule. Do not change existing rows.
- [ ] **Step 4:** Commit `results/M2.md` if it is tracked. Otherwise note in the task report that
  results/ is gitignored and the numbers live there locally.

### Task 2: M6 for all eight pitchers

**Files:**
- Modify: `scripts/m6_evaluate.py:26` (PITCHERS), `scripts/m6_location.py` (its pitcher list),
  `results/M6.md`, `src/pitchsim/sequence/m6_held_out.json` (via `scripts/m6_export_summary.py`)

- [ ] **Step 1:** Extend PITCHERS in both scripts to the eight pitchers in Global Constraints (display
  names: Ohtani, Misiorowski, Sale, Cease, Yamamoto, King, Sanchez, Schlittler).
- [ ] **Step 2:** Run `PYTHONPATH=src ../arkitecht/.venv/Scripts/python.exe scripts/m6_evaluate.py`,
  then `scripts/m6_location.py`. These are long runs; use run_in_background.
- [ ] **Step 3:** `scripts/m6_export_summary.py` reads `m6_transformer_folds.csv`. The transformer is
  **not** rerun (it needs torch and is not used by M7), so give that script a guard: a pitcher
  missing from the transformer CSV gets `"transformer": null`. Then run it.
- [ ] **Step 4:** Update `results/M6.md`: add the five pitchers to the log-loss and calibration
  tables, state whether "every model beats the count baseline on held-out games" still holds for
  all eight, and say that the transformer covers the original three only.
- [ ] **Step 5:** Run `PYTHONPATH=src ../arkitecht/.venv/Scripts/python.exe -m pytest tests/test_mcp_tools.py tests/test_sequence_*.py -q`.
  Expected: PASS. (`mcp_tools` reads `m6_held_out.json`; fix the reader only if a null breaks it.)
- [ ] **Step 6:** Commit the scripts, `m6_held_out.json` and `results/M6.md` (if tracked):
  `git commit -m "M6 for all eight viewer pitchers"`.

### Task 3: The average hitter

**Files:**
- Create: `src/pitchsim/sequence/hitter.py`, `scripts/m7_hitter.py`, `tests/test_sequence_hitter.py`

**Interfaces:**
- Produces:
  - `OUTCOMES = ["take", "whiff", "foul", "in_play"]`
  - `outcome_of(description: str) -> str | None`
  - `zone_bin(x_ft, z_ft, sz_top_ft, sz_bot_ft) -> str` (vectorised over arrays)
  - `class HitterModel(C: float = 1.0)` with `.fit(df) -> HitterModel`,
    `.predict(df) -> np.ndarray (n, 4)` and `.to_json() -> dict`
  - `class HitterTable(a: float = 1.0)`: the count-by-zone-bin baseline, with the same methods.
  - `df` needs columns `count`, `stand`, `pitch_type`, `x_ft`, `z_ft`, `sz_top`, `sz_bot`,
    `description`.

Why it exists: spec goal 4. The model is fitted per pitcher, on that pitcher's real pitch results
against both sides.

- [ ] **Step 1: Write the failing tests**

```python
def test_outcome_mapping():
    assert outcome_of("ball") == "take" and outcome_of("called_strike") == "take"
    assert outcome_of("blocked_ball") == "take" and outcome_of("pitchout") == "take"
    assert outcome_of("swinging_strike") == "whiff" and outcome_of("swinging_strike_blocked") == "whiff"
    assert outcome_of("foul_tip") == "whiff"      # a caught foul tip is a strike that can end the AB
    assert outcome_of("missed_bunt") == "whiff"
    assert outcome_of("foul") == "foul" and outcome_of("foul_bunt") == "foul"
    assert outcome_of("hit_into_play") == "in_play"
    assert outcome_of("hit_by_pitch") is None     # dropped: no HBP in M7 (spec non-goal)

def test_zone_bins():
    top, bot = 3.5, 1.6
    assert zone_bin(0.0, 2.5, top, bot) == "heart"
    assert zone_bin(0.70, 2.5, top, bot) == "edge_in"     # 0.13 ft inside the 0.83 ft half-width
    assert zone_bin(0.95, 2.5, top, bot) == "edge_out"
    assert zone_bin(1.3, 2.5, top, bot) == "chase"
    assert zone_bin(2.5, 2.5, top, bot) == "waste"

def test_predict_rows_sum_to_one_and_fit_signal():   # synthetic frame built in the test, labelled SYNTHETIC
    df = synthetic_hitter_frame(n=4000, seed=0)      # in-zone pitches: swing far more often than waste ones
    p = HitterModel().fit(df).predict(df)
    assert np.allclose(p.sum(1), 1)
    heart = (zone_bin(df.x_ft, df.z_ft, df.sz_top, df.sz_bot) == "heart").to_numpy()
    assert p[heart, 0].mean() < p[~heart, 0].mean()  # fewer takes in the heart
```

- [ ] **Step 2:** Run `PYTHONPATH=src ../arkitecht/.venv/Scripts/python.exe -m pytest tests/test_sequence_hitter.py -q`.
  Expected: FAIL (module missing).
- [ ] **Step 3: Implement `hitter.py`.**
  - **Zone:** half-width = 17/24 ft + ball radius 1.45/12 ft ≈ 0.829 ft. Vertical range is
    `[sz_bot − r, sz_top + r]`. Signed distance from the zone edge is in inches, negative inside.
  - **Bins:** heart < −3 in, edge_in −3..0, edge_out 0..3, chase 3..9, waste > 9.
  - **HitterModel:** a multinomial logistic on one-hot (count, stand, pitch_type, zone_bin), the
    same pipeline style as `logistic.py`. `to_json()` writes encoder categories, coefficients,
    intercepts and class order.
  - **HitterTable:** P(outcome | count, zone_bin), Dirichlet-smoothed toward the pitcher's
    overall outcome mix (like `CountBaseline`).
- [ ] **Step 4:** Run the tests. Expected: PASS.
- [ ] **Step 5: `scripts/m7_hitter.py`:**
  - For each of the 8 pitchers, run 5 game folds (`validation.game_folds`). Tune C and `a` on a
    30% slice of the training games, as `m6_evaluate.tune` does.
  - Report held-out log loss for the model vs the table, overall and per stand.
  - Write `results/m7/hitter.csv` with columns pitcher, stand, n_test, ll_model, ll_table,
    folds_better, use.
  - `use` = "model" where ll_model < ll_table on held-out pitches, otherwise "table" (spec
    ruling 4).
  - Pitch location is the front-of-plate crossing from `m6_location.py`'s helper; import it, do
    not copy it.
- [ ] **Step 6:** Run the script and check the CSV has 16 rows (8 pitchers × 2 stands).
- [ ] **Step 7:** Commit `hitter.py`, the test and the script:
  `git commit -m "M7 average hitter: take/whiff/foul/in-play model with count-by-zone baseline"`.

### Task 4: Aiming, the bank and the re-aim check

**Files:**
- Create: `src/pitchsim/face.py`, `tests/test_face.py`

**Interfaces:**
- Consumes:
  - `generative.fit_pitch_type`, `sample` and `generated_release_state`;
  - `seam.fit_pitch_types` and `predict_push`;
  - `physics.simulate` and `PhysicsParams`;
  - `export._sampled` (knots), `KNOT_S`.
- Produces:
  - `aim_velocity(rel_pos_ft, speed_fps, spin_rpm, axis_deg, params, push, target_xz_ft) -> np.ndarray (3,)`:
    the release velocity whose M1 + M3 flight crosses y = 17/12 ft at target (x, z), within 0.01 in.
  - `reaim_knots(sim: dict, dx_ft: float, dz_ft: float) -> dict`: the browser's approximation, as
    the Python reference.
  - `make_bank(train, rho, fits, loc_means, n_per_type=200, seed=0) -> dict[str, list[dict]]`.
    `loc_means[type]` = that type's unconditional location mean (ft). Each entry holds:
    - `sim` (`{"s", "T"}` knots);
    - `plate` [x, z, T] and `aim` [x, z];
    - `release` [x, y, z];
    - `speed_mph`, `spin_rpm`, `spin_axis_deg`, `extension_ft`;
    - `break_in` [x, z] (vs a no-spin flight, as in the export).
  - `reaim_error_in(bank, targets_by_type, params_by_type) -> pd.DataFrame`: per pitch, the plate
    distance in inches between `reaim_knots` and a true re-simulation aimed with `aim_velocity`.

How the pieces fit (spec ruling 1):
- Each bank pitch's own M2 release angles are discarded.
- The pitch is aimed with `aim_velocity` at its type's mean location.
- The browser later shifts it from that aim to the sampled target.

- [ ] **Step 1: Write the failing tests**

```python
def test_aim_hits_target():
    p = PhysicsParams(cd=0.33, spin_efficiency=0.9, rho=0.0740)
    v = aim_velocity(np.array([-1.8, 54.0, 5.9]), 140.0, 2300.0, 210.0, p, np.zeros(2), (0.3, 2.2))
    tr = simulate(np.array([-1.8, 54.0, 5.9]), v, 2300.0, 210.0, p)
    x, z = plate_crossing(tr)                       # helper in face.py: crossing of y = 17/12
    assert abs(x - 0.3) * 12 < 0.01 and abs(z - 2.2) * 12 < 0.01
    assert np.isclose(np.linalg.norm(v), 140.0)

def test_reaim_zero_offset_is_identity():
    sim = {"s": [[0, 54, 6, 0, -140, -5], [0.1, 1.5, 2.3, 1, -128, -15]], "T": 0.42}
    assert reaim_knots(sim, 0.0, 0.0) == sim

def test_reaim_moves_plate_end_exactly_and_release_not_at_all():
    out = reaim_knots(sim, 0.5, -0.25)
    assert out["s"][0][:3] == sim["s"][0][:3]
    assert np.allclose(np.array(out["s"][-1][:3]) - sim["s"][-1][:3], [0.5, 0, -0.25])
```

- [ ] **Step 2:** Run `PYTHONPATH=src ../arkitecht/.venv/Scripts/python.exe -m pytest tests/test_face.py -q`.
  Expected: FAIL.
- [ ] **Step 3: Implement.**
  - **`aim_velocity`:** solve the two release angles with `scipy.optimize.least_squares` on the
    plate miss. The starting guess is a straight line to the target.
  - **`reaim_knots`:**
    - Let u = (y₀ − y)/(y₀ − y_end) along the knots. Add u·(dx, dz) to x and z.
    - Add (dx, dz)·(−vy)/(y₀ − y_end) to vx and vz, so the Hermite tangents agree with the shifted
      positions.
    - `y` and `T` are unchanged. Docstring: a small release-angle change is, to first order, a
      shift that grows linearly with distance from release.
  - **`make_bank`:**
    - Draw M2 samples per type and get release position and speed from
      `generated_release_state`.
    - The push is `predict_push` on the sample (seam features are release-only, so this is valid).
    - Use the type's calibrated cd and efficiency from `fits`, and the training games' median rho.
- [ ] **Step 4:** Run the tests. Expected: PASS.
- [ ] **Step 5:** Add a test for Review Focus 4, `test_far_target_reaims_to_exact_plate_point`:
  - `dx = 3.0` ft gives a finite `s` and an end x equal to the old end x + 3.0;
  - `T` is unchanged.
- [ ] **Step 6:** Commit:
  `git commit -m "M7 aiming: solve release angles to a target, bank of aimed M2 pitches, re-aim reference"`.

### Task 5: Face export (Python → JSON) and the re-aim gate

**Files:**
- Create: `scripts/m7_export.py`, `tests/test_m7_export.py`
- Write: `web/public/data/face/<slug>.json` (8 files), `web/test/fixtures/face-synthetic.json`,
  `results/m7/reaim.csv`

**Interfaces:**
- Consumes: Tasks 3 and 4, `LogisticSeq(features=FEATURES + ["tto"])` and `LocationModel`.
- Produces the JSON layout, schema `face-1`, read by Task 6:

```
{ "meta": {"schema": "face-1", "synthetic": bool, "label": "Simulated (M2 + M1/M3, picked by M6)",
           "seed": 0, "knot_s": 0.05, "season": int, "train_games": int},
  "pitcher": {"name", "id", "throws"},
  "types": {"<code>": {"name", "arm_angle_deg", "n_train"}},    // only types with a bank
  "excluded": [{"type", "reason"}],
  "seq": {"features": [...], "categories": {feat: [levels]}, "classes": [...],
          "coef": [[...]], "intercept": [...], "floor": 1e-4},
  "location": {"<code>": {"features": [...], "categories": {...}, "coef": [[...],[...]],
                          "intercept": [x, z], "cov": [[a,b],[b,c]], "aim": [x, z]}},
  "hitter": {"<stand>": {"kind": "model" | "table", ...to_json()}},
  "zone": {"L": {"top_ft", "bot_ft"}, "R": {"top_ft", "bot_ft"}},  // training-game medians per side
  "bank": {"<code>": [bank entries from Task 4, floats rounded to 4 dp]},
  "parity": [...]   // only in the synthetic fixture: contexts with Python's exact outputs
}
```

- [ ] **Step 1: Write the failing tests**, run on the synthetic fixture produced by
  `scripts/m7_export.py --synthetic data/synthetic.csv --out web/test/fixtures/face-synthetic.json`:

```python
def test_layout_and_label():
    d = json.load(open(FIX))
    assert d["meta"]["schema"] == "face-1" and d["meta"]["synthetic"] is True
    assert set(d["seq"]["classes"]) == set(d["bank"]) == set(d["types"]) == set(d["location"])

def test_rare_type_absent_from_bank_is_never_in_seq_classes():
    d = json.load(open(FIX))
    for e in d["excluded"]:
        assert e["type"] not in d["seq"]["classes"] and e["type"] not in d["bank"]

def test_synthetic_may_not_leave_fixtures(tmp_path):
    with pytest.raises(SystemExit):
        main(["--synthetic", "data/synthetic.csv", "--out", str(tmp_path / "x.json")])

def test_size_budget():   # real exports, skipped if not yet written
    for f in Path("web/public/data/face").glob("*.json"):
        assert len(gzip.compress(f.read_bytes())) <= 1_000_000
```

- [ ] **Step 2:** Run them. Expected: FAIL.
- [ ] **Step 3: Implement `scripts/m7_export.py`.**
  - **Arguments:** `slug ...`, or `--all` for the 8 pitchers, with `--synthetic` for the fixture.
  - **Fitting:** fit everything on training games. A type goes in `excluded` with a reason if M2
    skipped it (MIN_TRAIN), M3 has no fit for it, or it has fewer than 30 located training
    pitches.
  - **seq:** refit LogisticSeq on the bank types only, so `classes == bank types`. C is tuned as
    in `m6_evaluate.tune`.
  - **location:** `LocationModel` per type. `aim` is the type's unconditional mean.
  - **hitter:** model or table per stand, from `results/m7/hitter.csv` `use`. Refit on the
    training games.
  - **parity:** 20 contexts. For each, write Python's type probabilities, location mean, hitter
    probabilities, and `reaim_knots` output for one bank pitch with a fixed (dx, dz).
  - **Re-aim gate:**
    - For each pitcher, call `reaim_error_in` on 50 bank pitches per type × targets sampled from
      that type's location model across all count/side contexts (spec ruling 2: the full range).
    - Write `results/m7/reaim.csv` with pitcher, type, n, median_in, p95_in and max_in.
    - **If any max_in > 0.5, stop and report BLOCKED with the table.** The controller rules on the
      fallback; do not invent one.
- [ ] **Step 4:** Run the tests. Expected: PASS. Then run `--all` and confirm the 8 files exist and
  pass the size budget.
- [ ] **Step 5:** Commit the script, the test, the fixture and `web/public/data/face/*.json`:
  `git commit -m "M7 export: per-pitcher face JSON (models + aimed pitch bank), re-aim check"`.

### Task 6: The JS at-bat engine

**Files:**
- Create: `web/src/face/engine.js`, `web/test/face-engine.test.js`

**Interfaces:**
- Consumes: `face-1` JSON (Task 5).
- Produces (all pure, no DOM):
  - `rng(seed: number) -> () => number` (mulberry32)
  - `typeProbs(face, ctx) -> {type: p}`
    - `ctx = {count: "b-s", stand: "L"|"R", prevType: string, tto: "1"}`;
    - one-hot over `seq.categories` (unknown level → all zeros), softmax, floor, renormalise over
      `seq.classes`.
  - `locationMean(face, type, ctx) -> [x, z]` and `sampleTarget(face, type, ctx, r) -> [x, z]`
    (Cholesky of `cov` × two Box–Muller normals)
  - `hitterProbs(face, stand, ctx, type, [x, z]) -> {take, whiff, foul, in_play}`
  - `isStrike(face, stand, [x, z]) -> boolean` (same zone rule as Task 3)
  - `reaim(sim, dx, dz) -> sim` (port of `reaim_knots`)
  - `newAtBat(face, {stand, batter: "none"|"avg", count: "0-0"}) -> ab`
    - `ab = {id, stand, generated: true, pitches: [], balls, strikes, event: null}`.
  - `throwPitch(face, ab, r) -> {ab, pitch}`
    - `pitch` uses the viewer's pitch fields (`pitch_type`, `pitch_name`, `speed_mph`, `spin_rpm`,
      `spin_axis_deg`, `arm_angle_deg`, `extension_ft`, `release`, `sz_top_ft`, `sz_bot_ft`,
      `sim`, `plate_sim`, `balls`, `strikes`, `description`, `status: "ok"`, `real: null`), plus
      `generated: true`, `outcome` and `why: {p, count, stand}`.
    - `tunnel.break_x_in` and `tunnel.break_z_in` come from the bank entry.
    - `description` uses Statcast words (`ball`, `called_strike`, `swinging_strike`,
      `foul_tip`, `foul`, `hit_into_play`) so the existing HUD colours apply.
  - `nextCount(ab, outcome, strike) -> {balls, strikes, event}`
    - `event` is `"strikeout"`, `"walk"`, `"in_play"` or `null`.

Bank draw: uniform over the type's bank (r), then `reaim` from the entry's `aim` to the sampled
target.

- [ ] **Step 1: Write the failing tests**

```js
const face = JSON.parse(readFileSync("test/fixtures/face-synthetic.json"));

test("parity with Python: type probs, location mean, hitter probs, reaim", () => {
  for (const c of face.parity) {
    for (const [t, p] of Object.entries(c.type_probs)) assert.ok(Math.abs(typeProbs(face, c.ctx)[t] - p) < 1e-9);
    assert.ok(dist(locationMean(face, c.type, c.ctx), c.loc_mean) < 1e-9);
    // hitter and reaim likewise, tolerance 1e-9 / 1e-6 ft
  }
});
test("typeProbs renormalises over bank types", () => {
  const p = typeProbs(face, { count: "0-0", stand: "R", prevType: "start", tto: "1" });
  assert.deepEqual(Object.keys(p).sort(), Object.keys(face.bank).sort());
  assert.ok(Math.abs(Object.values(p).reduce((a, b) => a + b) - 1) < 1e-12);
});
test("sampled types match typeProbs (chi-square, 20k draws, seed 1)", () => { /* p > 0.001 */ });
test("sampled targets match mean and cov (20k draws, 3 se)", () => {});
test("two-strike foul keeps count, foul tip strikes out", () => {
  assert.deepEqual(nextCount({ balls: 1, strikes: 2 }, "foul", true), { balls: 1, strikes: 2, event: null });
  assert.equal(nextCount({ balls: 1, strikes: 2 }, "whiff", true).event, "strikeout");
});
test("ball four walks, called third strike strikes out, in play ends", () => {});
test("batter none: every result is a take", () => {});
test("far target re-aims to the exact plate point", () => { /* dx = 3 ft: finite, end x shifted by 3 */ });
test("same seed, same at-bat", () => {});
test("generated pitch is labelled and has no real path", () => {
  const { pitch } = throwPitch(face, newAtBat(face, { stand: "R", batter: "none" }), rng(1));
  assert.equal(pitch.generated, true); assert.equal(pitch.real, null);
});
```

  "Foul tip" in the hitter output: when the outcome is `whiff` with 2 strikes, use description
  `foul_tip` with probability equal to the training share of foul tips among two-strike whiffs.
  Export that share as `hitter[stand].foul_tip_share` (add it to Task 5's output if missing).
  Either way it is a strikeout.
- [ ] **Step 2:** Run `cd web; npm test`. Expected: the new file FAILS, everything else passes.
- [ ] **Step 3:** Implement `engine.js`.
- [ ] **Step 4:** Run `npm test`. Expected: PASS.
- [ ] **Step 5:** Commit: `git commit -m "M7 engine: sampled at-bats from face JSON, parity-tested against Python"`.

### Task 7: At-bat checks (simulated vs real held-out at-bats)

**Files:**
- Create: `scripts/m7_real_stats.py`, `scripts/m7_check.mjs`, `results/M7.md` (draft; tracked
  only if results/ is tracked)

**Interfaces:**
- `m7_real_stats.py` writes `results/m7/<slug>-real.json`, computed on held-out games, per stand:
  - `n_ab`, `stand_share`;
  - `mix_by_count` {count: {type: share}};
  - `zone_rate`, `first_pitch_strike`, `pitches_per_ab`;
  - `outcomes` {strikeout, walk, in_play}.

  Each stat has a 95% interval from 2,000 game-level bootstrap resamples (seed 0). HBP at-bats are
  excluded and counted.
- `node scripts/m7_check.mjs [slug|--all]` imports `web/src/face/engine.js`. It:
  - simulates 20,000 Average-hitter at-bats per pitcher, with stand drawn from `stand_share` and
    seed 0;
  - computes the same statistics;
  - writes `results/m7/<slug>-sim.json` and `results/m7/checks.csv` (pitcher, stand, stat, real,
    lo, hi, sim, inside).
  - For `mix_by_count`, it reports the total-variation distance per count against the real
    bootstrap TV distribution's 95th percentile. `m7_real_stats.py` also exports that percentile.

- [ ] **Step 1:** Write `tests/test_m7_real_stats.py`. On a 3-game synthetic frame built in the
  test, check:
  - `pitches_per_ab` and `outcomes` equal hand-counted values;
  - every interval contains its point estimate.
  Run it and expect FAIL. Implement, then expect PASS.
- [ ] **Step 2:** Run both scripts for `--all`.
- [ ] **Step 3:** Draft `results/M7.md`:
  - what was measured;
  - the checks table (rows per pitcher × stand; ✓ where sim is inside the real interval);
  - the hitter table (Task 3);
  - the re-aim table (Task 5);
  - the excluded types;
  - "what's still wrong".

  The Batter-none mode has no real counterpart. Say so; do not grade it.
- [ ] **Step 4:** Commit the scripts and test (and `M7.md` if tracked):
  `git commit -m "M7 checks: simulated vs real held-out at-bats"`.

### Task 8: Mockups for Ryan (gate)

**Files:**
- Create (scratchpad, not committed): three PNGs from a throwaway `?debug=facemock` page or static
  HTML using the real `.gp` panel CSS.

- [ ] **Step 1:** Mock up, at desktop width and at phone width (390 px):
  - (a) the start-screen Replay / Face him choice and the Face him control row:
    - side L/R;
    - Batter: None / Average hitter;
    - Call the pitches;
    - start count;
    - New at-bat;
    - pace.
  - (b) the reveal card plus the at-bat strip, placed off to the side and translucent like the
    bottom-right panel, over a real batter-view frame.
  - (c) the guess buttons (Ball / Strike + optional type chips) and the score.
- [ ] **Step 2:** **Stop.** The controller shows Ryan the images and waits for his OK or changes
  before Task 9. Record his decisions in the ledger.

### Task 9: Viewer integration (Face him mode)

**Files:**
- Modify: `web/src/state.js`, `web/src/main.js`, `web/src/data.js`, `web/src/menu-ui.js` (start
  choice), `web/index.html`, `web/src/ui.css`, `web/vercel.json` (no change expected: `/data/`
  already covers `face/`)
- Test: `web/test/state.test.js` (add), `web/test/face-state.test.js`

**Interfaces:**
- `data.js`: `loadFace(slug) -> Promise<face>`, memoised like `loadPitcher`.
- `state.js`:
  - New fields: `face: null | {data, ab, rng, opts: {stand, batter, call, startCount, paceS}}`
    and `kind: "replay" | "face"`.
  - New actions: `enterFace {face}`, `leaveFace`, `faceOpts {...}`, `facePitch` (throws the next
    pitch with the engine and appends it), `faceNewAtBat`.
  - `currentAtBat` / `currentPitch` return the face at-bat and pitch when `kind === "face"`.
  - `shownPaths` returns `[p.sim]` for `p.generated`, whatever `mode` is: there is no real path,
    and the Real/Sim/Ghost buttons are disabled in Face him.
  - On `leaveFace`, the stored replay `abIndex` / `pitchIndex` / `t` are restored exactly.
- `main.js`:
  - The camera side follows `face.opts.stand`.
  - The pitcher's delivery uses the generated pitch's `release` and `arm_angle_deg` through the
    same `pitcherApi` call as real pitches. Ohtani's signed-off path is unchanged (spec ruling 3).
  - The HUD shows a "Simulated" tag on every generated pitch.
  - After each pitch lands, auto-advance after `paceS` (default 2.5 s). When the at-bat ends,
    wait for New at-bat, or auto-start after 4 s if nothing was touched.
- **URL:** `?face=1&stand=L|R&batter=none|avg&call=0|1` deep-links into Face him. Keyboard: `F`
  toggles Replay/Face him (check that no existing key uses F; if one does, pick a free key and note
  it).
- Pitchers with no face file (a load failure) get a card from the existing `showCard` saying Face
  him isn't available for him, and stay in Replay.

- [ ] **Step 1: Write failing tests** in `face-state.test.js` using the synthetic fixture:
  - `enterFace` → `kind === "face"` and `currentPitch` is null until `facePitch`;
  - after `facePitch`, `shownPaths` has one path, and it is the generated `sim`;
  - **`switching back to replay restores ab and pitch`**: set replay to ab 3, pitch 2, t 0.1;
    enter, throw 2, leave → the same ab, pitch and t;
  - a stand change mid-at-bat starts a fresh at-bat at the start count;
  - Real/Sim/Ghost actions are ignored while `kind === "face"`.
- [ ] **Step 2:** Run `npm test`. Expected: the new tests FAIL.
- [ ] **Step 3:** Implement `state.js` and `data.js`, then wire `main.js`, `menu-ui.js`, `index.html`
  and `ui.css` following the approved mockups (Task 8 ledger entry).
- [ ] **Step 4:** Run `npm test`. Expected: all PASS.
- [ ] **Step 5:** Headless screenshots on the dev server (copy and adapt the scratchpad
  `fans\shot.ps1`; the dev server on port 5240 stays up):
  - Face him batter view for a righty and a lefty;
  - Ohtani in Face him;
  - phone width.

  Look at them. Confirm the delivery releases at the generated release point, and that Replay
  looks identical to before (compare against a pre-change screenshot of the same URL).
- [ ] **Step 6:** Commit: `git commit -m "M7 viewer: Face him mode plays generated at-bats"`.

### Task 10: Reveal card, at-bat strip, call-the-pitches

**Files:**
- Create: `web/src/face/ui.js`, `web/src/face/guess.js` (pure scoring), `web/test/face-guess.test.js`
- Modify: `web/src/main.js`, `web/src/ui.css`

**Interfaces:**
- `guess.js`:
  - `scoreCall(call: "ball"|"strike"|null, pitch) -> "right"|"wrong"|"skipped"`. The truth is
    `isStrike` at the plate (the zone call, not the hitter outcome).
  - `scoreType(pick: string|null, pitch) -> "right"|"wrong"|"skipped"`.
  - `tally(scores) -> {calls: [right, total], types: [right, total]}` (skipped is not counted in
    totals).
- `ui.js`:
  - `showReveal(pitch)` renders the card per spec goal 6, in the approved position. It fades in for
    3 s and never blocks input.
  - `why` reads "1-2 vs RHB: 41% SL" from `pitch.why`.
  - The strip appends a chip per pitch (type, speed, result colour).
  - The guess UI appears at release while `call` is on and closes when the ball lands.
  - Keys `B` = ball and `S` = strike; type chips are clickable. Check for key conflicts as in
    Task 9.

- [ ] **Step 1: Write failing tests** in `face-guess.test.js`:
  - right, wrong, and **`no call scores as skipped`** (`scoreCall(null, p) === "skipped"`, and
    `tally` totals exclude it);
  - a pitch on the zone edge (Task 3's half-width exactly) is a strike.
- [ ] **Step 2:** Run, and expect FAIL. Implement `guess.js` and `ui.js`. Run `npm test`, and expect
  PASS.
- [ ] **Step 3:** Screenshots as in Task 9:
  - the reveal card with the call off;
  - the guess UI mid-flight;
  - the score after 5 pitches;
  - phone width.

  The card must not overlap the zone or the pitcher in the batter and umpire views. Check the
  bounding boxes in the screenshot against the projected zone.
- [ ] **Step 4:** Commit: `git commit -m "M7 reveal card, at-bat strip and call-the-pitches"`.

### Task 11: Finish

- [ ] **Step 1:** Run the Python suite in the background (about 20 min):
  `PYTHONPATH=src ../arkitecht/.venv/Scripts/python.exe -m pytest -q`. Run `cd web; npm test`.
  Both must pass.
- [ ] **Step 2:** Run `cd web; npx vite build`, then `npx vite preview --port 5250`. Check:
  - Face him for all 8 pitchers loads with no console errors;
  - `face/*.json` is served from `dist/data/face/`.
- [ ] **Step 3:** Measure frame time in Face him vs Replay (umpire view), the same way as
  `results/web-performance.md`. Add a row to that file.
- [ ] **Step 4:** Finish `results/M7.md`:
  - is each "done when" met;
  - the per-pitcher checks summary;
  - what's still wrong.
- [ ] **Step 5:** Commit. Do not push; the controller merges via PR after Ryan's OK.
