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
python scripts/validate.py data/669373_2025-04-01_2025-09-30.csv
```

`scripts/validate.py` prints a table and writes `results/summary.csv` and
`results/report.png`. No network? `python -m pitchsim.synthetic` makes a fake pitcher
with known physics so you can run the pipeline anyway (calibration should
recover Cd = 0.35 and roughly the spin efficiencies listed in `src/pitchsim/synthetic.py`).

## Files

| File | What it does |
|---|---|
| `src/pitchsim/physics.py` | Gravity + drag + Magnus force, RK4 integration. `simulate` (one pitch, full path) and `simulate_batch` (thousands at once, for fitting). Air density from temperature/elevation. |
| `src/pitchsim/statcast.py` | Turns Statcast's 9-parameter fit into a starting position/velocity at y = 50 ft. |
| `scripts/fetch.py` | Downloads a pitcher's pitches with `pybaseball`. |
| `scripts/validate.py` | The experiment: data check → convention check → default physics → calibrated physics → bias. |
| `src/pitchsim/synthetic.py` | Fake pitcher with known physics, for testing without the internet. |
| `tests/` | Checks against known numbers: ~0.4 s flight, ~8% speed loss, realistic rise, Coors effect. |

## How to read the output

- **[0] fit vs plate** should be near 0 inches. If not, coordinates are being read wrong.
- **[1] spin_axis vs break direction** should be a few degrees. Confirms the spin sign convention.
- **baseline_err_in** — median miss at the plate using textbook constants.
- **cd, spin_eff** — fitted per pitch type. Spin efficiency is the share of spin that
  actually makes the ball move. Expect ~0.85–1.0 for four-seamers, much lower for
  gyro-heavy sliders and cutters.
- **calibrated_err_in** — median miss on *held-out* pitches. This is your headline number.
- **bias_x_in / bias_z_in** — average signed miss. A consistent miss in one
  direction is movement spin can't explain. That's where step 3 (seam-shifted wake) starts.

## Honest limitations (= the roadmap)

1. **We start at y = 50 ft, not at release.** Next: start at the release point,
   so the simulator covers the first ~4 ft too, then generate pitches from
   release conditions alone.
2. **spin_axis in public Statcast is inferred from movement**, so it partly
   bakes the answer into the input. True "from scratch" prediction would use the
   *observed* spin axis; the gap between the two is the seam-shifted-wake signal.
3. **The constant-acceleration fit is itself an approximation.** On synthetic data
   it costs ~0.2–0.5 in and pulls fitted spin efficiencies slightly low, so treat
   anything under about half an inch as the noise floor.
4. **Air density is one value per run.** Real games vary (Coors Field, cold April
   nights). Statcast rows include the venue and date; joining weather would help.
5. **Cd is constant.** In reality it depends on speed and spin.

## References

- Alan Nathan's baseball physics site and trajectory calculator (the standard model used here).
- Baseball Savant's CSV documentation for column definitions.
- Barton Smith / Driveline research on seam-shifted wake, for step 3.
