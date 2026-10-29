# M1 completion: physics engine, validated on real pitchers

Date: 2026-10-29. Source of truth for the overall project: Ryan's project brief (pasted in the
design conversation). Step 1 saves it verbatim as `CLAUDE.md` at the repo root, so its
layout, dependency list and conventions are what "the brief" means below.

## Intent

The end goal is a **virtual Trajekt machine**: a gray 3D figure on a laptop that delivers a
specific MLB pitcher's pitches, flies them the way his pitches really move, and sequences them
the way he does. Ryan chose to build it milestone by milestone (M1 → M7), each part done
properly and measured before the next, rather than a rough end-to-end first.

M1 is the ball-flight foundation everything later sits on. It's done when, for real pitchers,
we can say in inches how far a simulated pitch lands from the real one, per pitch type, on
held-out pitches, and where the systematic misses are.

Ryan is building this to learn. Code must explain the *why* (physics, math, modelling choices)
in short comments/docstrings with references. Work proceeds in small steps with a plan before
each non-trivial step.

## Current state

`pitch-sim.zip` in the repo root holds ~715 lines of working Python (flat layout):
`physics.py` (RK4, gravity + drag + Magnus, `air_density(temp_f, elevation_ft)`),
`statcast.py` (9-param fit → state at y = 50 ft, recovered by running the fit back from
`release_pos_y`), `fetch_data.py` (pybaseball download), `validate.py` (data check →
convention check → default physics → per-pitch-type calibration of Cd and spin efficiency on
70% / held-out 30% → bias), `synthetic.py` (fake pitcher with known physics),
`tests/test_physics.py`. The repo is otherwise empty (README only).

## Goals

1. **Restructure** into the brief's layout: `src/pitchsim/` package (`physics.py`,
   `statcast.py`, `data.py` for fetch/cache/clean), `scripts/` for CLI entry points
   (`fetch.py`, `validate.py`, `synthetic.py` or equivalent), `tests/`, `pyproject.toml`
   (Python 3.11+, editable install). All existing tests pass unchanged in meaning.
   Unzip the archive into the new layout; the zip itself is not committed.
   `data/` and `results/` gitignored.
2. **Simulate from the release point**, not y = 50 ft. Initial state = Statcast's
   constant-acceleration fit evaluated at `release_pos_y` (position and velocity there).
   Report, as a data check, how far the fit's position at release is from the reported
   `release_pos_x/z` (expected ~0 since the y = 50 state is currently derived from those;
   say so in the report rather than presenting it as independent evidence).
   Keep y = 50 ft start available so both numbers can be compared in the write-up.
3. **Per-game air density.** Venue comes from the StatsAPI game feed (`gameData.venue`), not
   Statcast's `home_team`, so neutral sites and temporary homes (A's in Sacramento, Rays at
   Steinbrenner Field) come out right *(Claude's call)*. Elevation from a small hand-written
   table keyed by venue ID (unknown venue → sea level with a warning). Game-time temperature from MLB's public StatsAPI game feed
   (`game_pk` → `gameData.weather.temp`), cached on disk in `data/raw/` and never re-fetched.
   Missing temperature → 70 °F with a count reported. Roofed/dome games: use the reported
   temperature as-is (the feed reports indoor temp for closed roofs) and note it as a
   limitation. Only new dependency allowed here: `requests` (pybaseball already pulls it in).
   Calibration and evaluation use each pitch's own air density.
4. **Run on three real pitchers** with different arsenals: **Shohei Ohtani, Chris Sale,
   Jacob Misiorowski**, seasons **2025 + 2026** (regular season, same window for all three)
   *(Claude's call; Ryan moved on without objecting)*. Pitch types under the 30-pitch minimum
   are listed as skipped, not silently dropped. Raw downloads cached in `data/raw/`. Each pitcher's outputs go to
   their own folder, e.g. `results/m1/<pitcher>/summary.csv` + `report.png`
   (the current script overwrites `results/summary.csv`; fix that).
   **Held-out split is by game, not by pitch** *(Claude's call)*: within each pitcher, 70% of
   games for calibration and 30% held out, with a fixed seed. Pitches from one game share air
   conditions, so a pitch-level split would leak and flatter the numbers.
5. **Write-up** `results/M1.md`: per pitcher and pitch type, median and p90 held-out plate
   error (in), fitted Cd and spin efficiency, bias x/z table, plate-speed error. Compare
   against the old setup (y = 50 ft start, 70 °F, sea level), with each change reported
   **separately and combined**: release start only, per-game density only, both
   *(Claude's call)*. what's still wrong
   (expected: sinkers/changeups/sweepers bias → seam-shifted wake, M3). Every number in it is
   reproducible by running scripts in `scripts/`.

## Non-goals

- No M2+ work (generative model, seam model, viewer, pose, sequencing). No scaffolding for
  them either.
- No speed/spin-dependent Cd model (listed as a known limitation only).
- No humidity, wind, or barometric pressure.

## Anti-goals

- Synthetic results presented next to or mixed into real-pitcher results without a clear
  label. Synthetic stays a test/demo tool.
- Leakage: calibration must never see held-out pitches. Fixed seeds throughout.
- Re-downloading cached data.
- Heavy dependencies (anything beyond the brief's stack + `requests`) without asking.

## Constraints

- Coordinates and conventions per the brief (feet; x catcher's right, y plate→pitcher, z up;
  plate_x/z at y = 17/12 ft).
- Type hints, docstrings, units in names or docstrings.
- Small steps: each numbered goal above is its own reviewable step, in order, with Ryan's OK
  between steps.

## Left to the builder

- Exact module/function names inside `src/pitchsim/`, and whether `synthetic.py` lives in
  `src/` or `scripts/`.
- How the per-game weather fetch is batched and the cache file format.
- Plot design for per-pitcher reports, provided it follows the existing report's style.
