# Pitcher Replica — Project Brief

> Permanent context for an AI coding assistant working in this repo.
> Save as `CLAUDE.md` (Claude Code) or `.github/copilot-instructions.md` (Copilot).

## Who I am and how to work with me

I'm Ryan, a CS-track student at UBC. This is a personal project I'm building to **learn**
physics simulation, probabilistic ML, sequence models, 3D graphics, and computer vision. I care
about understanding the work more than about finishing fast.

- **Explain before you build.** For anything non-trivial, give me a short plan (what, why,
  which files) and wait for my OK.
- **Small steps.** One milestone sub-task at a time. Don't scaffold future milestones early.
- **Teach the why.** When you add physics, math, or a modelling choice, add a short comment or
  docstring explaining it, with a reference if there is one.
- **Never pass off fake data as real.** Synthetic data is fine for tests. It must be labelled
  as synthetic and kept apart from real-data results.
- **Every model gets a measurable check.** No milestone is done without a number that says how
  good it is, measured on held-out data.
- Ask before adding heavy dependencies (PyTorch, Unity, cloud services).

## The goal

Reproduce a specific MLB pitcher's pitches **as exactly as possible**:

1. **Flight:** how each pitch moves (gravity, drag, spin, seam effects)
2. **Look:** release point, extension, arm slot, approach angle, tunneling
3. **Variation:** pitch-to-pitch scatter, plus how it shifts with fatigue and count
4. **Delivery:** the body motion, recovered from video
5. **Sequencing:** what he throws next, sampled realistically (not just the most likely pitch)

**North-star test:** a simulated pitch should be statistically indistinguishable from a real one
by the same pitcher. Each milestone measures part of that.

## Data

- **Statcast** via `pybaseball` (`statcast_pitcher`): every MLB pitch. Key columns: `pitch_type`,
  `release_speed`, `release_pos_x/y/z`, `release_extension`, `vx0, vy0, vz0, ax, ay, az` (the
  9-parameter constant-acceleration fit, defined at y = 50 ft), `plate_x`, `plate_z`, `pfx_x`,
  `pfx_z`, `release_spin_rate`, `spin_axis`, `balls`, `strikes`, `outs_when_up`, `inning`,
  `stand`, `at_bat_number`, `pitch_number`, `game_pk`, `game_date`, `arm_angle` (if present).
- **Coordinates:** feet; x toward the catcher's right, y from the plate toward the pitcher, z up.
  In the 2026 CSVs, `plate_x/plate_z` are at mid-plate (y ≈ 0.71 ft), not the front (y = 17/12 ft);
  M1 scores against the 9-parameter fit evaluated at y = 17/12 instead.
- **Caveat:** public `spin_axis` is *Hawk-Eye-measured*, not inferred from movement (checked
  2026-10-30 against Savant's spin-direction leaderboard: within ~7° of "measured", 4–33° off
  "inferred", for 15 pitch types). The gap between it and the axis implied by movement is the
  seam-shifted-wake signal (Milestone 3), available per pitch.
- Cache raw downloads in `data/raw/` (gitignored). Never re-download what's cached.
- Video (Milestone 5) is for personal research only: don't commit or redistribute clips.

## Repo layout (target)

```
pitch-sim/
  src/pitchsim/
    physics.py        # flight model (M1)
    statcast.py       # Statcast parsing / coordinate helpers (M1)
    data.py           # fetching, caching, cleaning
    generative.py     # per-pitcher pitch distributions (M2)
    seam.py           # non-Magnus movement model (M3)
    tunneling.py      # look / deception metrics (M4)
    pose/             # video -> delivery kinematics (M5)
    sequence/         # pitch-selection models (M6)
    export.py         # JSON for the web viewer
  scripts/            # CLI entry points (fetch, validate, train, export)
  notebooks/          # exploration only; real logic lives in src/
  web/                # Three.js batter's-eye viewer (M4+)
  tests/
  data/  results/     # gitignored
  pyproject.toml
```

Python 3.11+, `numpy`, `pandas`, `scipy`, `matplotlib`, `pytest`. Add `scikit-learn` / `xgboost`
at M2/M6, `torch` at M6 (ask first), MediaPipe or RTMPose at M5, Three.js + Vite at M4.

## Milestones

### M1: Physics engine, validated (partly done)
Existing code: RK4 simulator with gravity + drag + Magnus, per-pitch-type calibration of Cd and spin
efficiency, held-out plate-error report, synthetic-pitcher test, physics sanity tests.
Remaining:
- Restructure into `src/pitchsim/` with `pyproject.toml`; keep all tests passing.
- Run on 2–3 real pitchers with different arsenals; record results in `results/`.
- Start simulations **from the release point** instead of y = 50 ft.
- Per-game air density from venue elevation (+ temperature if obtainable).
- **Done when:** median held-out plate error per pitch type is reported for real pitchers, and the
  per-pitch-type bias table is written up.

### M2: Per-pitcher generative model ("the arm")
- For each pitch type, model the joint distribution of release x/z, extension, velocity, spin rate,
  spin axis, spin efficiency. Start with a multivariate Gaussian, then a Gaussian mixture.
- Condition on context: pitch count in game (fatigue), times through the order, count.
- Sample → simulate with M1 physics → full synthetic pitches.
- **Done when:** a classifier two-sample test (train a model to tell real from simulated) is near
  chance (AUC close to 0.5), and marginal distributions match.

### M3: Seam-shifted wake
- Fetch observed vs inferred spin axis per pitcher/pitch type.
- Model the movement the Magnus force can't explain as a function of that gap (start linear).
- **Done when:** M1/M2 bias for affected pitches (sinkers, changeups, sweepers) shrinks measurably.

### M4: The look ("batter's eye")
- `tunneling.py`: approach angles (vertical/horizontal), separation between pitch pairs at the
  hitter's decision point (~23–25 ft from the plate), perceived velocity from extension.
- `export.py` → JSON trajectories; `web/` Three.js viewer from the batter's box at real timing,
  with release point, strike zone, and pitch selection.
- **Done when:** I can pick a pitcher and watch real vs simulated pitches side by side.

### M5: The moving pitcher (perception → animated 3D delivery)
Goal: a 3D pitcher in the viewer whose delivery moves like the real one, with the ball leaving
his hand exactly where Statcast says it does. MLB's skeletal tracking isn't public, so the motion
has to be recovered from video.
- **5a: 2D pose.** Run MediaPipe/RTMPose on center-field broadcast clips. Learn the failure
  modes: motion blur at release, occlusion, camera cuts.
- **5b: 3D motion.** Lift to 3D with a monocular human-mesh model (e.g. HMR 2.0 / 4D-Humans, or
  WHAM for world-grounded motion), smooth over time, and average several deliveries per pitch type.
- **5c: Anchor to Statcast.** Scale and position the skeleton with known field geometry (rubber,
  plate). Constrain the throwing hand at release to `release_pos_x/z` and `release_extension`,
  and the arm to `arm_angle`.
- **5d: Animate.** Retarget onto a rigged character in the Three.js viewer and sync release with
  the M1 ball flight.
- Measured metrics: arm slot, stride length, release timing, time the ball is hidden.
- **Done when:** recovered release point / arm slot are within a stated error of Statcast's for
  several pitchers, and the animation plays in sync with the simulated pitch.
- Same core skill as teaching humanoid robots from human video (motion capture → retargeting),
  so design `pose/` cleanly enough to reuse.

### M6: Sequencing ("the brain")
- Baselines: pitch-type frequency by count → logistic / XGBoost with context (count, handedness,
  previous pitch, score, runners) → small transformer over the at-bat pitch history.
- Also predict target location (pitch type → location distribution).
- **Split by game, not by pitch** (no leakage). Evaluate with log loss and calibration plots,
  not just accuracy. Always **sample** from the distribution.
- **Done when:** each model beats the count baseline on held-out games, and calibration is reported.

### M7: Integration: "Face the pitcher"
Simulated at-bats: M6 picks the pitch → M2 generates it → M1/M3 fly it → M4 shows it.

### Stretch
- WebXR mode in the viewer.
- Inverse problem: wheel speeds/angles for a pitching machine that reproduce a target pitch.

## Conventions
- Type hints and docstrings on public functions; units in names or docstrings (`_ft`, `_mph`, `_rpm`).
- Fixed random seeds. Every result is reproducible from a script in `scripts/`.
- Each milestone ends with a short `results/<milestone>.md`: what was measured, the numbers,
  what's still wrong.
- Tests for physics invariants and data parsing. Models are judged by their evaluation scripts.

## References
- Alan Nathan, baseball physics papers and trajectory calculator (flight model, lift/drag).
- Baseball Savant CSV documentation (column definitions).
- Barton Smith / Driveline research on seam-shifted wake.
- Pitch tunneling literature (Baseball Prospectus).
