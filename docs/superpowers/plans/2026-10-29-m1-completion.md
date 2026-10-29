# M1 Completion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the zipped flat-file physics engine into a `pitchsim` package that simulates from the release point with per-game air density, validate it on Ohtani (2025), Sale (2026) and Misiorowski (2026), and write up `results/M1.md`.

**Architecture:** Library code in `src/pitchsim/` (`physics`, `statcast`, `data`, `validation`, `synthetic`); thin CLIs in `scripts/` (`fetch.py`, `validate.py`). `validation.py` holds the experiment (split, calibrate, evaluate) so it is testable; `scripts/validate.py` only parses args, runs it over four variants, and draws the report.

**Tech Stack:** Python 3.11+, numpy, pandas, scipy, matplotlib, pytest, pybaseball. Game feed fetched with stdlib `urllib.request` + `json` (ponytail: no `requests` needed).

**Spec:** `docs/superpowers/specs/2026-10-29-m1-completion-design.md`

## Global Constraints

- Coordinates: feet; x catcher's right, y plate→pitcher, z up; `plate_x/plate_z` at y = 17/12 ft.
- Type hints + docstrings on public functions; units in names or docstrings (`_ft`, `_mph`, `_rpm`).
- Every physics/math/modelling choice gets a short *why* comment, with a reference where one exists (Ryan is learning from this code).
- Fixed seeds (`seed=0`). Calibration never sees held-out games.
- Raw downloads cached in `data/raw/`; never re-downloaded. `data/` and `results/` gitignored except `results/*.md`.
- No new dependencies beyond `pyproject.toml` below. No M2+ code or scaffolding.
- Synthetic data is labelled synthetic and never written under `results/m1/<real pitcher>/`.
- Each task is its own reviewable step; stop for Ryan's OK between tasks (CLAUDE.md: "small steps").

## Review Focus

1. Game feed has no `weather`, or `temp` is `""`/missing (postponed/suspended games) → 70 °F used, and the count of such games is printed.
2. Network dies halfway through fetching game conditions → rows fetched so far are already in the cache; rerun only fetches the rest.
3. Venue ID not in the elevation table (neutral site, London/Tokyo/Seoul series) → sea level, warning printed with venue name and ID.
4. Pitch type with ≥30 pitches but zero pitches in held-out games (or <30 total) → listed as skipped with its count, not silently dropped.
5. Name lookup with no match or a misspelling ("Misorowski") → clear error message, no empty CSV cached.

---

### Task 1: Restructure into `src/pitchsim/`

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `src/pitchsim/__init__.py` (empty)
- Move (from `pitch-sim.zip`, unzipped to a temp dir): `physics.py`, `statcast.py`, `synthetic.py` → `src/pitchsim/`; `fetch_data.py` → `scripts/fetch.py`; `validate.py` → `scripts/validate.py`; `tests/test_physics.py` → `tests/test_physics.py`; merge zip `README.md` into repo `README.md`
- Delete: `pitch-sim.zip` (untracked; delete after unzip, never commit it)
- Already present: `CLAUDE.md` (the brief, committed with this plan)

**Interfaces:**
- Produces: `pitchsim.physics`, `pitchsim.statcast`, `pitchsim.synthetic` importable after `pip install -e .`; intra-package imports become `from pitchsim.physics import ...`.

- [ ] **Step 1:** Unzip `pitch-sim.zip` into a temp directory outside the repo and `git mv`-equivalent copy files to the paths above. Remove the `sys.path.insert` hack from `tests/test_physics.py`; import `from pitchsim.physics import ...`. Scripts import from `pitchsim`. `synthetic.py` keeps its `__main__`, run as `python -m pitchsim.synthetic`, writing `data/synthetic.csv`.
- [ ] **Step 2:** `pyproject.toml`: setuptools, `src` layout, `requires-python = ">=3.11"`, dependencies `numpy, pandas, scipy, matplotlib, pybaseball>=2.2.7`, optional `dev = ["pytest"]`, `[tool.pytest.ini_options] testpaths = ["tests"]`.
- [ ] **Step 3:** `.gitignore`: `data/`, `results/*` with `!results/*.md`, `*.zip`, `__pycache__/`, `.venv/`, `*.egg-info/`.
- [ ] **Step 4:** Run `python -m venv .venv; .venv\Scripts\pip install -e .[dev]; .venv\Scripts\pytest -q` → 6 passed.
- [ ] **Step 5:** Smoke: `python -m pitchsim.synthetic` then `python scripts/validate.py data/synthetic.csv` → runs, calibrated Cd within 0.02 of 0.35.
- [ ] **Step 6:** Commit `"Restructure into src/pitchsim package"` (include `CLAUDE.md` if not yet committed).

---

### Task 2: Per-pitch air density in `simulate_batch`

**Files:**
- Modify: `src/pitchsim/physics.py` (`simulate_batch`)
- Test: `tests/test_physics.py`

**Interfaces:**
- Produces: `simulate_batch(pos0, vel0, spin_rpm, spin_axis_deg, cd, spin_efficiency, rho: float | np.ndarray | None = None, y_stop=FRONT_OF_PLATE_Y, dt=1e-3)`; `rho` may be scalar or length-N.

- [ ] **Step 1: Failing test**

```python
def test_batch_per_pitch_density():
    v = 80 * MPH * np.array([0.0, -1.0, 0.0])
    sea, coors = air_density(70, 0), air_density(70, 5200)
    pos, _, _ = simulate_batch(np.stack([RELEASE, RELEASE]), np.stack([v, v]),
                               [2600, 2600], [0, 0], 0.33, 0.9, rho=np.array([sea, coors]))
    for i, rho in enumerate([sea, coors]):
        one = simulate(RELEASE, v, 2600, 0, PhysicsParams(cd=0.33, spin_efficiency=0.9, rho=rho))
        assert np.allclose(pos[i], one["pos"][-1], atol=1e-6)
```

- [ ] **Step 2:** `pytest tests/test_physics.py::test_batch_per_pitch_density -v` → FAIL (shape error or mismatch).
- [ ] **Step 3:** Broadcast `k` to shape `(n,)` like `cd`, and index `k[m][:, None]` inside `f`.
- [ ] **Step 4:** `pytest -q` → 7 passed.
- [ ] **Step 5:** Commit `"simulate_batch: per-pitch air density"`.

---

### Task 3: Start simulations from the release point

**Files:**
- Modify: `src/pitchsim/statcast.py`
- Test: `tests/test_statcast.py` (new)

**Interfaces:**
- Consumes: `initial_state(df)` (unchanged, y = 50 ft), `_time_to_y`.
- Produces: `release_state(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]` — position (N×3, = `release_pos_x/y/z`) and velocity (N×3) of Statcast's fit evaluated at `release_pos_y`.

- [ ] **Step 1: Failing tests** (use `pitchsim.synthetic.make(n=300, seed=0)`, labelled synthetic in the test docstring)

```python
def test_release_state_position_is_release_point():
    df = make(n=300, seed=0)
    pos, _ = release_state(df)
    assert np.allclose(pos, df[["release_pos_x", "release_pos_y", "release_pos_z"]].to_numpy())

def test_release_state_velocity_runs_fit_backwards():
    """Release speed > speed at y=50 (drag slows the ball), by under ~2 mph over ~5 ft."""
    df = make(n=300, seed=0)
    _, v_rel = release_state(df)
    _, v_50 = initial_state(df)
    dv = (np.linalg.norm(v_rel, axis=1) - np.linalg.norm(v_50, axis=1)) * 0.681818
    assert np.all(dv > 0) and np.all(dv < 2)

def test_release_start_recovers_synthetic_plate_location():
    """Simulating from release with the TRUE physics lands near plate_x/z.
    Residual is the constant-acceleration fit's own approximation error."""
    df = make(n=300, seed=0)
    pos0, vel0 = release_state(df)
    pos, _, _ = simulate_batch(pos0, vel0, df["release_spin_rate"], df["spin_axis"],
                               TRUE_CD, df["true_spin_eff"])
    err_in = np.hypot(pos[:, 0] - df["plate_x"], pos[:, 2] - df["plate_z"]) * 12
    assert np.median(err_in) < 1.0
```

- [ ] **Step 2:** `pytest tests/test_statcast.py -v` → FAIL (`release_state` not defined).
- [ ] **Step 3:** Implement `release_state` in `statcast.py`: `t_rel = _time_to_y(release_pos_y, vy0, ay)`; velocity = `v0 + a * t_rel`. Docstring explains: Statcast's fit is defined at y = 50 ft but valid over the whole flight, so evaluating it at release recovers the release state; the position equals `release_pos_*` *by construction* (we derive the y = 50 position from it), so it is not independent evidence.
- [ ] **Step 4:** `pytest -q` → all pass. If the third test fails, report the median rather than loosening the bound silently.
- [ ] **Step 5:** Commit `"Add release_state: simulate from the release point"`.

---

### Task 4: Fetch, cache, and per-game conditions (`data.py`)

**Files:**
- Create: `src/pitchsim/data.py`
- Modify: `scripts/fetch.py` (thin CLI over `data.fetch_pitcher`)
- Test: `tests/test_data.py`

**Interfaces:**
- Produces:
  - `RAW_DIR: Path` = repo-root `data/raw`
  - `fetch_pitcher(name: str | None, mlbam_id: int | None, start: str, end: str) -> Path` — cached CSV `data/raw/<id>_<start>_<end>.csv`, regular season only (`game_type == "R"`); returns existing file without network if present; raises `SystemExit("No MLB player found for '<name>'")` on no match and never writes an empty file.
  - `parse_game_feed(feed: dict) -> dict` → keys `game_pk: int, venue_id: int, venue_name: str, temp_f: float` (`nan` if weather/temp absent or not numeric).
  - `fetch_game_conditions(game_pks: Iterable[int], cache_path: Path = RAW_DIR / "game_conditions.csv") -> pd.DataFrame` — fetches `https://statsapi.mlb.com/api/v1.1/game/{pk}/feed/live` only for pks not in the cache; appends each row to the cache as it's fetched.
  - `VENUE_ELEVATION_FT: dict[int, tuple[str, float]]` — venue_id → (name, elevation ft).
  - `add_air_density(df: pd.DataFrame, conditions: pd.DataFrame) -> pd.DataFrame` — adds `rho` (lb/ft³) per pitch via `physics.air_density(temp_f, elevation_ft)`; temp NaN → 70 °F; unknown venue → 0 ft; prints counts of each fallback and the unknown venue names/IDs.

- [ ] **Step 1: Failing tests**

```python
COORS = {"gamePk": 1, "gameData": {"venue": {"id": 19, "name": "Coors Field"},
                                   "weather": {"temp": "72", "condition": "Clear"}}}

def test_parse_game_feed():
    assert parse_game_feed(COORS) == {"game_pk": 1, "venue_id": 19,
                                      "venue_name": "Coors Field", "temp_f": 72.0}

def test_parse_game_feed_missing_weather():
    feed = {"gamePk": 2, "gameData": {"venue": {"id": 19, "name": "Coors Field"}}}
    assert np.isnan(parse_game_feed(feed)["temp_f"])
    feed["gameData"]["weather"] = {"temp": ""}
    assert np.isnan(parse_game_feed(feed)["temp_f"])

def test_add_air_density_uses_venue_and_temp():
    pitches = pd.DataFrame({"game_pk": [1, 2, 3]})
    cond = pd.DataFrame({"game_pk": [1, 2, 3], "venue_id": [19, 19, 999999],
                         "venue_name": ["Coors Field", "Coors Field", "Nowhere"],
                         "temp_f": [72.0, np.nan, 60.0]})
    out = add_air_density(pitches, cond)
    coors_ft = VENUE_ELEVATION_FT[19][1]
    assert out["rho"].tolist() == pytest.approx([air_density(72, coors_ft),
                                                 air_density(70, coors_ft),
                                                 air_density(60, 0)])

def test_coors_is_high():
    assert 5150 < VENUE_ELEVATION_FT[19][1] < 5250

def test_fetch_game_conditions_uses_cache(tmp_path, monkeypatch):
    cache = tmp_path / "gc.csv"
    pd.DataFrame([parse_game_feed(COORS)]).to_csv(cache, index=False)
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: pytest.fail("network hit"))
    assert fetch_game_conditions([1], cache).loc[0, "venue_id"] == 19
```

- [ ] **Step 2:** `pytest tests/test_data.py -v` → FAIL (module missing).
- [ ] **Step 3:** Implement `data.py`. Move `lookup_id` and download logic from `scripts/fetch.py` into `fetch_pitcher`; `scripts/fetch.py` keeps the arg parsing and summary print. Elevation table: all 30 current parks plus Sutter Health Park (Sacramento) and George M. Steinbrenner Field (Tampa); verify each venue ID against the feed while running Task 6 and add any missing ones that appear. Comment the table with the source used for elevations. Use `urllib.request.urlopen(url, timeout=30)` + `json.load`.
- [ ] **Step 4:** `pytest -q` → all pass.
- [ ] **Step 5:** Commit `"data.py: cached fetch, game venue/temperature, per-game air density"`.

---

### Task 5: Validation by game, four variants, per-pitcher output

**Files:**
- Create: `src/pitchsim/validation.py` (move `run_sim`, `residuals`, `calibrate`, `angle_diff`, and the per-pitch-type loop out of `scripts/validate.py`)
- Modify: `scripts/validate.py` (CLI + `make_report` only), `src/pitchsim/synthetic.py` (add `game_pk`)
- Test: `tests/test_validation.py`

**Interfaces:**
- Consumes: `release_state`, `initial_state`, `simulate_batch(rho=array)`, `fetch_game_conditions`, `add_air_density`.
- Produces:
  - `split_by_game(df: pd.DataFrame, train_frac: float = 0.7, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]`
  - `evaluate(df: pd.DataFrame, start: Literal["y50", "release"], rho: np.ndarray) -> tuple[pd.DataFrame, list[str]]` — per-pitch-type rows with the existing summary columns; second item lists skipped types as `"<type> (n=<count>): <reason>"`.
  - CLI: `python scripts/validate.py <csv> [--out results/m1/<slug>] [--no-weather]`. Runs variants `y50+const`, `release+const`, `y50+game`, `release+game` (const = `air_density(70, 0)`); writes `<out>/summary.csv` (long format, `variant` column first), `<out>/report.png` (from `release+game`), prints skipped types. `--no-weather` skips the `+game` variants (synthetic data). Default `--out` is `results/m1/<player_name slug>`; refuses to write synthetic data (`"Synthetic" in player_name`) anywhere except `results/synthetic/`.

- [ ] **Step 1: Failing tests** (synthetic data)

```python
def test_split_by_game_has_no_shared_games():
    df = make(n=600, seed=0)
    train, test = split_by_game(df)
    assert set(train.game_pk).isdisjoint(test.game_pk)
    assert len(train) + len(test) == len(df)

def test_split_by_game_is_deterministic():
    df = make(n=600, seed=0)
    assert split_by_game(df)[1].index.equals(split_by_game(df)[1].index)

def test_evaluate_recovers_true_cd_from_release():
    df = make(n=1500, seed=0)
    rows, skipped = evaluate(df, "release", np.full(len(df), air_density()))
    assert rows["cd"].between(0.33, 0.37).all()
    assert (rows["calibrated_err_in"] < 1.0).all()

def test_evaluate_lists_rare_pitch_types_as_skipped():
    df = make(n=1500, seed=0)
    df = df[(df.pitch_type != "CU") | (df.index < df[df.pitch_type == "CU"].index[10])]
    _, skipped = evaluate(df, "release", np.full(len(df), air_density()))
    assert any(s.startswith("CU") for s in skipped)
```

- [ ] **Step 2:** `pytest tests/test_validation.py -v` → FAIL.
- [ ] **Step 3:** Add `game_pk` to `synthetic.make` (`rng.integers(1, 41)` per pitch: 40 fake games). Implement `validation.py`; `split_by_game` shuffles unique `game_pk` with `np.random.default_rng(seed)` and puts the first `ceil(0.7 * n_games)` in train. The `start` argument picks `release_state` vs `initial_state`; `rho` array is subset alongside train/test. Keep `MIN_PITCHES = 30`, `MAX_TRAIN = 400`. Docstring on `split_by_game` explains the leakage reason (shared per-game air and conditions).
- [ ] **Step 4:** `pytest -q` → all pass; `python -m pitchsim.synthetic; python scripts/validate.py data/synthetic.csv --no-weather` → writes `results/synthetic/summary.csv`.
- [ ] **Step 5:** Commit `"Validation: split by game, four variants, per-pitcher results"`.

---

### Task 6: Real pitchers and `results/M1.md`

**Files:**
- Create: `results/M1.md`
- Modify: `src/pitchsim/data.py` (`VENUE_ELEVATION_FT` additions only, if warnings appear), `README.md` (run commands)

- [ ] **Step 1:** Fetch (regular season only, filtered by `game_type`):
  `python scripts/fetch.py --name "Shohei Ohtani" --start 2025-03-01 --end 2025-11-30`;
  `python scripts/fetch.py --name "Chris Sale" --start 2026-03-01 --end 2026-11-30`;
  `python scripts/fetch.py --name "Jacob Misiorowski" --start 2026-03-01 --end 2026-11-30`. Expected: three CSVs in `data/raw/`; rerunning prints "cached" and makes no request.
- [ ] **Step 2:** `python scripts/validate.py data/raw/<id>_<start>_<end>.csv` for each. Expected: `results/m1/<slug>/summary.csv` + `report.png` per pitcher; checks [0] and [1] near 0; fallback counts printed. Fix any unknown-venue warnings by adding to `VENUE_ELEVATION_FT` and rerun.
- [ ] **Step 3:** Write `results/M1.md`: what was measured (held-out *games*, 70/30, seed 0); per pitcher × pitch type table of n, median and p90 error, Cd, spin efficiency, bias x/z, plate-speed error for `release+game`; a small table of median error per variant showing what release start and per-game air each changed; skipped pitch types; data caveats (release position equals `release_pos_*` by construction; fit approximation floor ~0.2–0.5 in; roof games use reported temp; `spin_axis` is inferred); what's still wrong (biases pointing to M3). Every number copied from `summary.csv`, none hand-computed. Add the run commands to `README.md`.
- [ ] **Step 4:** `pytest -q` → all pass.
- [ ] **Step 5:** Commit `"M1 results: Ohtani, Sale, Misiorowski"` (M1.md + README + any table additions).
