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
on decimals alone). On the same three pitchers, 1 of 9 graded pitch types passes (AUC <= 0.55);
the other 8 fail at 0.63 to 0.73 (Gaussian). A real-vs-real floor (real training-game pitches
vs real held-out ones, same test) is 0.60 to 0.68 in 8 of those 9 rows, and the Gaussian is within
0.05 of it in 8 of 9; only Misiorowski's slider (0.729 vs floor 0.599) is clearly above it. So
most of the gap looks like game-to-game drift (suggested, not proven). The numbers are in
[`results/M2.md`](results/M2.md). To reproduce it (cached CSVs from the M1 step):

```bash
python scripts/train_generative.py data/raw/660271_2025-03-01_2025-11-30.csv
python scripts/train_generative.py data/raw/519242_2026-03-01_2026-11-30.csv
python scripts/train_generative.py data/raw/694819_2026-03-01_2026-11-30.csv
```

This writes `results/m2/<pitcher>/` (summary, KS and context CSVs, one PNG per pitch type).

## Files

| File | What it does |
|---|---|
| `src/pitchsim/physics.py` | Gravity + drag + Magnus force, RK4 integration. `simulate` (one pitch, full path) and `simulate_batch` (thousands at once, for fitting). Air density from temperature/elevation. |
| `src/pitchsim/statcast.py` | Turns Statcast's 9-parameter fit into a starting position/velocity at release (`release_state`) or y = 50 ft (`initial_state`), and gives the fit's own plate crossing. |
| `scripts/fetch.py` | Downloads a pitcher's pitches with `pybaseball`. |
| `scripts/validate.py` | The experiment: data check → convention check → default physics → calibrated physics → bias. |
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
  direction is movement spin can't explain. That's where M3 (seam-shifted wake) starts.

## Honest limitations (= the roadmap)

1. **The release start is anchored on Statcast's fit.** The simulator now starts at the
   release point, but the release position is `release_pos_*` and the velocity is the fit
   extrapolated back. That is not an independent measurement of release.
2. **Public `spin_axis` is Hawk-Eye-measured, not inferred from movement.** Confirmed 2026-10-30
   against Savant's spin-direction leaderboard: within ~7° of "measured", 4–33° off "inferred",
   across 15 pitch types (see `CLAUDE.md`). Its gap to the movement-implied axis is the
   seam-shifted-wake signal, M3's job.
3. **The constant-acceleration fit is itself an approximation.** On synthetic data
   it costs ~0.2–0.5 in and pulls fitted spin efficiencies slightly low, so treat
   anything under about half an inch as the noise floor.
4. **Air density is per game, but only from elevation and temperature.** No humidity,
   wind or pressure, and roofed games use the reported (indoor) temperature.
5. **Cd is constant.** In reality it depends on speed and spin.

## References

- Alan Nathan's baseball physics site and trajectory calculator (the standard model used here).
- Baseball Savant's CSV documentation for column definitions.
- Barton Smith / Driveline research on seam-shifted wake, for M3.
