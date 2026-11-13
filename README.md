# pitch-sim — step 1: a physics engine validated against Statcast

Goal of the whole project: reproduce a specific MLB pitcher's pitches exactly —
flight, look, variation, sequencing. Step 1 is the foundation: **a pitch-flight
simulator, and a number that says how close it gets to real pitches.**

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .[dev]
pytest -q                                            # physics sanity checks
```

## Run it

```bash
# 1. Pull a season of real pitches (Baseball Savant, free)
python scripts/fetch.py --name "Tarik Skubal" --start 2025-04-01 --end 2025-09-30

# 2. Validate the physics against them
python scripts/validate.py data/raw/669373_2025-04-01_2025-09-30.csv
```

`scripts/validate.py` prints a table and writes `results/m1/<pitcher>/summary.csv` and
`report.png`. No network? `python -m pitchsim.synthetic` makes a fake pitcher
with known physics so you can run the pipeline anyway (calibration should
recover Cd = 0.35 and roughly the spin efficiencies listed in `src/pitchsim/synthetic.py`).

## M1 results

Three real pitchers, regular season only. The write-up is in [`results/M1.md`](results/M1.md).
To reproduce it:

```bash
python scripts/fetch.py --name "Shohei Ohtani" --start 2025-03-01 --end 2025-11-30
python scripts/fetch.py --name "Chris Sale" --start 2026-03-01 --end 2026-11-30
python scripts/fetch.py --name "Jacob Misiorowski" --start 2026-03-01 --end 2026-11-30
python scripts/validate.py data/raw/660271_2025-03-01_2025-11-30.csv
python scripts/validate.py data/raw/519242_2026-03-01_2026-11-30.csv
python scripts/validate.py data/raw/694819_2026-03-01_2026-11-30.csv
```

## M2 results

A per-pitcher generative model of how a pitch leaves the hand (release point, extension,
velocity, spin rate, spin axis), scored by a classifier that tries to tell real held-out pitches
from generated ones (generated values are rounded like Statcast's first, or the classifier wins
on decimals alone). Pass means the model's AUC is within a noise-scaled margin of a
real-vs-real floor (real training-game pitches vs real held-out ones, same test; 0.60 to 0.68 in
8 of 9 graded rows, so a fixed 0.55 bar is unreachable). On the same three pitchers, 6 of 9
graded Gaussian pitch types pass; the fails are Misiorowski's slider (gap +0.130 vs margin 0.040),
and two near misses, Sale's changeup (+0.046 vs 0.043) and slider (+0.021 vs 0.020). Under the
old AUC <= 0.55 bar only 1 of 9 passes. The numbers, including a physical-units table
(mean and spread differences in ft, mph, rpm, degrees), are in [`results/M2.md`](results/M2.md). To reproduce it (cached CSVs from the M1 step):

```bash
python scripts/train_generative.py data/raw/660271_2025-03-01_2025-11-30.csv
python scripts/train_generative.py data/raw/519242_2026-03-01_2026-11-30.csv
python scripts/train_generative.py data/raw/694819_2026-03-01_2026-11-30.csv
```

This writes `results/m2/<pitcher>/` (summary, KS, context and physical-units CSVs, one PNG per pitch type).

## M3 results

A learned constant push added to M1's flight, for the movement spin can't explain, on five
pitchers (the three above plus Cristopher Sánchez and Michael King, 2026). 14 of 17 graded pitch
types pass (median held-out miss drops by at least 0.6 in, 95% interval above zero); the fails are
Misiorowski's four-seamer and slider and King's slider. Graded medians go from 2.03–8.26 in to
1.49–3.15 in, 9 of 17 at or under 2 in and none at or under 1 in. The data mostly shows a fixed habit
per pitch type rather than a per-pitch effect (per-pitch beats fixed in 6 of 17), and the push isn't
shown to be seam force: it also holds a release-state artefact, and M1's efficiency can absorb much
of a push on the spin-force line. M1's efficiency lower bound moved from 0.20 to 0.01 during M3;
the old bound gave 16 of 17, but two of those passes were mostly repairing M1. The numbers and
synthetic controls are in
[`results/M3.md`](results/M3.md). To reproduce it:

```bash
python scripts/fetch.py --id 650911 --start 2026-03-01 --end 2026-11-30
python scripts/fetch.py --id 650633 --start 2026-03-01 --end 2026-11-30
python scripts/train_seam.py data/raw/<pitcher>.csv   # each of the five CSVs
python scripts/seam_controls.py                       # SYNTHETIC controls
python scripts/seam_bound_check.py                    # real data, old 0.20 bound vs new default
python scripts/seam_split_check.py                    # real data, four different game splits
```

This writes `results/m3/<pitcher>/` (summary and knock-on CSVs, one PNG per pitch type).

## M4: the batter's-eye viewer

**Status: done, pending Ryan's live-fps check.** M4 is a Three.js viewer of a stylised, code-built
ballpark. You pick a pitcher and one of his held-out at-bats, then watch each pitch fly in one of
three ways: as Statcast's fit, as M1 + M3, or both at once (ghost mode). There are five views
(umpire, batter, side, ride, free), a trail and a no-spin reference line. A skinned pitcher's hand
sits on the real release point.

Checks, all from [`results/M4.md`](results/M4.md):

- **Truth test:** the viewer reproduces the exported paths within 0.0025 in, over 3,400 real
  pitches.
- **Perceived velocity:** within 0.21–0.32 mph of Statcast's `effective_speed` per pitcher, on
  held-out games. This is after one reference extension was fitted on training games.
- **Approach angles:** no public Savant reference was found, so they rest on hand-worked tests.
- **Tunneling math:** unit-tested.

To run it:

```bash
python scripts/export_viewer.py data/raw/<pitcher>.csv [more.csv ...] --out web/public/data
cd web && npm install && npm run dev      # open the printed URL; press ? for keys
npm test                                  # truth and framing tests
python scripts/m4_checks.py data/raw/<pitcher>.csv [...]   # writes results/m4/checks.csv
```

Pass all five CSVs to `export_viewer.py` in one run: it rewrites `web/public/data/index.json`
each time. The exported JSON is gitignored.

## Files

| File | What it does |
|---|---|
| `src/pitchsim/physics.py` | Gravity + drag + Magnus force, RK4 integration. `simulate` (one pitch, full path) and `simulate_batch` (thousands at once, for fitting). Air density from temperature/elevation. |
| `src/pitchsim/statcast.py` | Turns Statcast's 9-parameter fit into a starting position/velocity at release (`release_state`) or y = 50 ft (`initial_state`), and gives the fit's own plate crossing. |
| `scripts/fetch.py` | Downloads a pitcher's pitches with `pybaseball`. |
| `scripts/validate.py` | The experiment: data check → convention check → default physics → calibrated physics → bias. |
| `src/pitchsim/seam.py`, `scripts/train_seam.py` | M3: learn the extra push per pitch type and per pitch, score it on held-out games. |
| `scripts/seam_controls.py` | M3 controls on synthetic pitches: what the method finds when the truth is known. |
| `scripts/seam_bound_check.py` | Record of the efficiency bound change: real results at the old 0.20 vs the new 0.01, plus the Spearman figures. |
| `scripts/seam_split_check.py` | M3: pass counts when a different set of games is held out (split seeds 0-3). |
| `src/pitchsim/tunneling.py`, `src/pitchsim/export.py`, `scripts/export_viewer.py` | M4: approach angles, perceived velocity, tunnel separation; held-out at-bats as JSON for the viewer. |
| `scripts/m4_checks.py` | M4 checks: perceived velocity vs `effective_speed`, approach angles per pitch type. |
| `web/` | M4 viewer (Three.js + Vite): cameras, HUD, pitcher figure, starter park in `web/src/park/`. |
| `src/pitchsim/synthetic.py` | Fake pitcher with known physics, for testing without the internet. |
| `tests/` | Checks against known numbers: ~0.4 s flight, ~8% speed loss, realistic rise, Coors effect. |

## How to read the output

- **[0] fit vs plate** should be near 0 inches. If not, either coordinates are being read wrong
  or Statcast reports plate_x/z at a different y. 2026 data reports them mid-plate
  (y = 0.708 ft, also printed), not the front edge. Scoring uses the fit's own crossing,
  so results don't depend on it.
- **[1] spin_axis vs break direction** should be a few degrees. Confirms the spin sign convention.
- **baseline_err_in** — median miss at the plate using textbook constants.
- **cd, spin_eff** — fitted per pitch type. Spin efficiency is the share of spin that
  actually makes the ball move. Expect ~0.85–1.0 for four-seamers, much lower for
  gyro-heavy sliders and cutters.
- **calibrated_err_in** — median miss on *held-out* pitches. This is your headline number.
- **bias_x_in / bias_z_in** — average signed miss. A consistent miss in one
  direction is movement spin can't explain. That's what M3 (`results/M3.md`) adds a push for.

## Honest limitations (= the roadmap)

1. **The release start is anchored on Statcast's fit.** The simulator now starts at the
   release point, but the release position is `release_pos_*` and the velocity is the fit
   extrapolated back. That is not an independent measurement of release.
2. **Public `spin_axis` is Hawk-Eye-measured, not inferred from movement.** Confirmed 2026-10-30
   against Savant's spin-direction leaderboard: within ~7° of "measured", 4–33° off "inferred",
   across 15 pitch types (see `CLAUDE.md`). M3 reports its gap to the movement-implied axis
   next to the learned push; the two rise together only loosely.
3. **The constant-acceleration fit is itself an approximation.** On synthetic data
   it costs ~0.2–0.5 in and pulls fitted spin efficiencies slightly low, so treat
   anything under about half an inch as the noise floor.
4. **Air density is per game, but only from elevation and temperature.** No humidity,
   wind or pressure, and roofed games use the reported (indoor) temperature.
5. **Cd is constant.** In reality it depends on speed and spin.
6. **M3's push is "what's left after M1", not measured seam force.** It includes a release-state
   artefact (up to about 1 ft/s² on synthetic pitches), and M1's spin efficiency can absorb much of
   a push on the spin-force line, as far as it can move inside its 0.01–1.0 bounds, so a small
   residual doesn't prove there's no seam effect.

## References

- Alan Nathan's baseball physics site and trajectory calculator (the standard model used here).
- Baseball Savant's CSV documentation for column definitions.
- Barton Smith / Driveline research on seam-shifted wake, for M3.
