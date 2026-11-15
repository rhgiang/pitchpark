# M5: The Moving Pitcher Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ohtani's real bases-empty delivery is recovered from Savant clips and played on the viewer figure, so the ball leaves his hand at the Statcast release point in sync with the M1 flight. Held-out accuracy and smoothness are reported, and an eye-test page compares the replica with a real clip.

**Architecture:** A new `src/pitchsim/pose/` package does the following in the main venv:
- clip lookup and download;
- 2D pose and screening;
- release-frame detection;
- averaging, anchoring and metrics.

Only the 3D lift needs PyTorch. It runs as a separate script in `.venv-pose` and writes `.npz` files that the main venv reads, so the main venv never imports torch. `scripts/m5_build.py` turns the clips into one delivery JSON (24 SMPL joints, positions per frame) plus metrics. The viewer plays it on a timeline that now starts before release (t < 0).

**Tech Stack:**
- Main venv: Python 3.12, numpy, pandas, scipy. The 2D pose tool is the builder's choice (Task 2), lightweight and ONNX/CPU-friendly.
- `.venv-pose`: PyTorch with a CUDA 12.8+ build, plus the 3D lifter chosen in Task 3.
- Viewer: Three.js 0.180.0 and Vite 7.3.8; tests with `node --test test/*.test.js`.

**Spec:** `docs/superpowers/specs/2026-11-13-m5-moving-pitcher-design.md`

## Global Constraints

- **No video in git, ever.**
  - Clips and every frame or image derived from them live under `data/video/` (gitignored by `data/`).
  - Delivery and clip-index JSON hold numbers only.
  - Clips are for personal research only (CLAUDE.md).
- **No `Co-Authored-By` or any Claude attribution in any commit or PR**, including subagent commits.
- **PyTorch only in `.venv-pose`** (CUDA 12.8+ build, RTX 5060 Ti / Blackwell):
  - `pip install -e .` and the existing tests must not need torch, video libraries or a GPU;
  - nothing under `src/pitchsim/` imports torch.
- **Season and split:**
  - Ohtani 2026 only (`data/raw/660271_2026-03-01_2026-11-30.csv`): M2's "latest season" rule.
  - Bases empty means `on_1b`, `on_2b` and `on_3b` are all NaN.
  - Split with `validation.split_by_game(df, seed=0)` on the cleaned 2026 rows, then filter to bases empty.
  - Clips from held-out games never build the delivery.
- **Metrics are measured before** the mean-release correction and before any per-pitch pinning.
- **Units in names** (`_ft`, `_s`, `_deg`). Statcast feet: x toward the catcher's right, y from the plate tip toward the pitcher, z up.
- **Field facts:**
  - the rubber's front edge is at y = 60.5 ft;
  - the mound top is 10/12 ft above the plate;
  - `release_extension` = 60.5 − release y.
- **Arm angle** follows Statcast's definition: the angle of the throwing shoulder → hand line at release above horizontal (0° sidearm, 90° straight over the top).
- **Synthetic test data is labelled synthetic** and never written next to real results.
- **Fixed seeds; every result is reproducible from `scripts/`.** `results/M5.md` closes the milestone.
- **Network politeness:** cache every feed lookup and every clip. Never re-download what's cached. Use the generic `User-Agent: Mozilla/5.0` and send nothing personal.
- **Node 24:** run web tests as `node --test test/*.test.js`.

## Review Focus

1. **A pitch whose release is far from his mean** (0.8 ft off, a real outlier) must still be reached by the arm without the elbow locking straight or the hand jumping. Test in Task 7.
2. **A pitcher with no delivery file** (the other five, a 404) must behave exactly as in M4: t starts at 0, the static release pose shows, and nothing negative leaks into the URL or the cameras. Test in Task 6.
3. **The spoiler rule during the windup:** at t < 0 the pitch has not landed. A mode switch, pitch change or seek to a negative t must not reveal the result. Test in Task 6.
4. **Clip download failures** (a feed without `playId`, a page with no mp4, an HTTP error) are recorded as dropped with a reason, and the run continues. Test in Task 1.
5. **A clip that is too short** to cover the delivery window around release is dropped with the reason "short", not padded or crashed on. Test in Task 4.

---

### Task 1: Clips (pitch rows → playId → cached mp4)

**Files:**
- Create: `src/pitchsim/pose/__init__.py` (empty), `src/pitchsim/pose/clips.py`
- Create: `scripts/m5_fetch_clips.py`
- Test: `tests/test_pose_clips.py`, fixtures `tests/fixtures/feed_min.json` (a hand-trimmed live feed: two plays with pitch events, labelled synthetic in a `"_note"` key) and `tests/fixtures/sporty_page.html` (a minimal page with one mp4 link)

**Interfaces:**
- Produces:
  - `bases_empty(df: pd.DataFrame) -> pd.Series[bool]`
  - `play_ids(df: pd.DataFrame, cache_path: Path = RAW_DIR / "play_ids.csv") -> pd.Series` (index aligned to df; NaN where none). Columns cached: `game_pk, at_bat_number, pitch_number, play_id`.
  - `play_ids_from_feed(feed: dict) -> pd.DataFrame` (same columns; pure, for tests)
  - `clip_url_from_page(html: str) -> str | None`
  - `clip_name(row) -> str` = `f"{game_pk}_{at_bat_number}_{pitch_number}.mp4"`
  - `fetch_clips(df, out_dir: Path) -> pd.DataFrame` with columns `game_pk, at_bat_number, pitch_number, play_id, clip_path, status` (status ∈ `ok`, `cached`, `no_play_id`, `no_mp4`, `http_error`)

- [ ] **Step 1: Write the failing tests.**
  - `test_bases_empty`: three rows (none on; a runner on 1b; a runner on 3b) give `[True, False, False]`.
  - `test_play_ids_from_feed`: the fixture gives the right play_id for (at bat 1, pitch 2). Only `isPitch` events count, so a pickoff event in between doesn't shift the numbering.
  - `test_clip_url_from_page`: the fixture returns its `https://sporty-clips.mlb.com/...mp4` link; `"<html></html>"` returns None.
  - `test_fetch_clips_records_failures`: monkeypatch the two network helpers so that:
    - one row has no play_id;
    - one row's page has no mp4;
    - one row raises `urllib.error.HTTPError`;
    - one row succeeds.

    Assert the four statuses and that the function returned instead of raising.
- [ ] **Step 2:** Run `.venv/Scripts/python.exe -m pytest tests/test_pose_clips.py -v`. Expected: FAIL (module missing).
- [ ] **Step 3: Implement `clips.py`.**
  - The feed URL is `data.FEED_URL`; plays are `liveData.plays.allPlays[at_bat_number - 1]`.
  - Each pitch is the `pitch_number`-th event with `isPitch` true; its `playId` is the id.
  - Fetch feeds once per game, and append to the cache CSV as you go (same pattern as `data.fetch_game_conditions`).
  - The page is `https://baseballsavant.mlb.com/sporty-videos?playId=<id>`.
  - Write each mp4 to `out_dir / clip_name(row)` and skip any file that already exists (status `cached`).
- [ ] **Step 4:** Run the tests. Expected: PASS.
- [ ] **Step 5: Add `scripts/m5_fetch_clips.py`.**
  - Arguments: `CSV --max-train 120 --max-test 80 --seed 0 --out data/video/<slug>`.
  - Split the cleaned rows by game with `split_by_game(seed=0)`, keep bases-empty rows, and sample up to the caps with `numpy.random.default_rng(seed)`.
  - Write `<out>/clips.csv`: the fetch table plus `split` (train/test), `pitch_type`, `release_pos_x`, `release_pos_z`, `release_extension`, `arm_angle` and `p_throws`.
  - Print counts by status.
  - Run it for Ohtani and check that `clips.csv` has around 200 rows with mostly `ok` or `cached`.
- [ ] **Step 6:** Commit `src/pitchsim/pose/`, `scripts/m5_fetch_clips.py`, the tests and the fixtures. Do not commit `data/`.

### Task 2: 2D pose, screening and release frame

**Files:**
- Create: `src/pitchsim/pose/pose2d.py`, `src/pitchsim/pose/screen.py`, `src/pitchsim/pose/release.py`
- Create: `scripts/m5_pose2d.py`, `scripts/m5_label_frames.py`
- Create: `src/pitchsim/pose/labels/ohtani_release_frames.csv` (committed: frame numbers only)
- Modify: `pyproject.toml` (an optional extra `video` for the 2D tool and video reading; not in the base install)
- Test: `tests/test_pose_release.py`

**Interfaces:**
- Consumes: `clips.csv` from Task 1.
- Produces:
  - `Pose2D` dataclass with fields `xy: np.ndarray (F, 17, 2)` in pixels, `conf: np.ndarray (F, 17)`, `fps: float`, `size: tuple[int, int]`. Keypoints use the COCO-17 order; the docstring lists it.
  - `load_pose2d(path) -> Pose2D`, `save_pose2d(p, path)` (npz, cached at `data/video/<slug>/pose2d/<clip>.npz`).
  - `screen(p: Pose2D, throws: str) -> tuple[bool, str]`, reason `""` when ok. Reasons: `no_pitcher`, `camera_cut`, `not_center_field`, `low_confidence`, `short`. The rules are the builder's; each gets a one-line "why" comment.
  - `release_frame(p: Pose2D, throws: str) -> int`

- [ ] **Step 1: Pick the 2D tool and add it to the `video` extra.**
  - Prefer RTMPose via `rtmlib` (ONNX) or MediaPipe.
  - Choose the one that tracks the pitcher (the largest person near frame centre) reliably on 5 Ohtani clips. Record the choice and why in the `pose2d.py` docstring.
- [ ] **Step 2: Write the failing tests** on a synthetic Pose2D (labelled synthetic). The throwing wrist moves slowly, peaks in speed at frame 40, then slows; fps is 60.
  - `test_release_frame_peak`: `release_frame` is within 1 frame of 40.
  - `test_release_frame_lefty`: the same motion on the left wrist with `throws="L"` gives 40, and the right wrist is ignored.
  - `test_screen_camera_cut`: the person's bounding box jumps by more than half the frame between two frames, giving `(False, "camera_cut")`.
  - `test_screen_short`: 20 frames at 60 fps gives `(False, "short")`.
- [ ] **Step 3:** Run them; FAIL. Implement; PASS.
- [ ] **Step 4: Add `scripts/m5_pose2d.py <clips.csv>`.** It runs 2D on every ok clip (cached), screens each one, finds the release frame, and writes `pose2d.csv` (clip, ok, reason, release_frame, fps, n_frames). It prints the yield by reason.
- [ ] **Step 5: Hand labels.**
  - `scripts/m5_label_frames.py` writes JPEG contact sheets around each detected release (±6 frames, frame numbers burned in) to `data/video/<slug>/labels/`. These are images, so they stay in `data/`.
  - The builder looks at the sheets for 20 clips spread across games and pitch types. Mark release as the last frame with the ball still in the hand, or the first frame where it is visibly out if the hand is blurred, and record which rule was used.
  - Write `ohtani_release_frames.csv` (clip, release_frame_label, rule, note).
  - The script then prints the detector's frame error on those 20 (median and maximum, in frames).
- [ ] **Step 6:** Commit the code, labels CSV, tests and pyproject.

### Task 3: PyTorch environment and the 3D lift (spike first)

**Files:**
- Create: `scripts/setup_pose_env.ps1` (creates `.venv-pose`, installs a CUDA 12.8+ PyTorch build and the chosen lifter, and prints `torch.cuda.get_device_name(0)`)
- Create: `scripts/m5_lift.py` (runs ONLY in `.venv-pose`; may import torch)
- Create: `src/pitchsim/pose/joints3d.py` (main venv: load and save the lift output; no torch)
- Create: `results/m5/spike.md` (gitignored folder; the findings are copied into `results/M5.md` in Task 9)
- Modify: `.gitignore` (add `.venv-pose/` and any lifter checkout or weights folder, e.g. `third_party/`)
- Test: `tests/test_pose_joints3d.py`

**Interfaces:**
- Produces:
  - Per clip, `data/video/<slug>/lift/<clip>.npz` with `joints_m: (F, 24, 3)` float32 (SMPL joint order exactly as `web/src/skeleton.js` `JOINTS`), `fps: float`, `up: (3,)` (the gravity-up unit vector in the same frame; world-grounded models give it, per-frame models estimate it from the feet over the clip), and `lifter: str`.
  - `load_lift(path) -> Lift3D` with fields `joints_m`, `fps`, `up`, `lifter`.

- [ ] **Step 1: The spike.** Try the lifters in this order and stop at the first one that installs and runs on the RTX 5060 Ti on 3 clips:
  1. GVHMR
  2. WHAM
  3. 4D-Humans (HMR 2.0)

  Record each attempt in `spike.md`: versions, the error if it failed, and seconds per clip.
- [ ] **Step 2: STOP for Ryan if the lifter needs body-model files** (SMPL, SMPL-X). Tell him the exact site, which files to download, and where to put them. These need a free registration under a non-commercial research licence, and he does it himself. Continue once they're in place.
- [ ] **Step 3: Write the failing test** `test_load_lift_roundtrip`. Save a synthetic `Lift3D` (24 joints, 10 frames, labelled synthetic), load it back, and check the shapes, `fps` and `up`. Also check that `load_lift` raises `ValueError` when the joint count isn't 24.
- [ ] **Step 4:** Implement `joints3d.py`; PASS.
- [ ] **Step 5: Implement `scripts/m5_lift.py <pose2d.csv>`.**
  - Lift every screened-ok clip (cached), mapping the lifter's output to the 24 SMPL joints.
  - SMPL `left_hand` and `right_hand` are the knuckle joints, which is where the ball sits at release.
  - Write a sanity check: median body height in metres over the clip, printed per clip.
  - Run it on all ok clips.
- [ ] **Step 6:** Commit the setup script, `m5_lift.py`, `joints3d.py`, the test and `.gitignore`.

### Task 4: One delivery: align, average, smooth, anchor, correct

**Files:**
- Create: `src/pitchsim/pose/delivery.py`, `src/pitchsim/pose/anchor.py`
- Test: `tests/test_pose_delivery.py`, `tests/test_pose_anchor.py`

**Interfaces:**
- Consumes: `Lift3D`, release frames.
- Produces:
  - `landmarks(j: np.ndarray, release: int, throws: str) -> dict[str, int]`, giving frame indices for `leg_lift` (peak height of the lead knee), `foot_strike` (the lead ankle's speed drops below 10% of its peak after leg_lift) and `release`. The lead leg is the glove side: left for a RHP.
  - `average_delivery(seqs: list[np.ndarray], marks: list[dict], fps_out: float = 60.0, pre_s: float, post_s: float = 0.6) -> tuple[np.ndarray, int]` returns (joints (T, 24, 3), release_index). It time-warps each clip piecewise-linearly so leg_lift, foot_strike and release line up at their median times, then resamples to `fps_out`, averages, and smooths with a Savitzky–Golay filter (`scipy.signal.savgol_filter`; window and order are the builder's, with a why-comment). A clip that doesn't cover [release − pre_s, release + post_s] is excluded with the reason `short`.
  - `jitter(j: np.ndarray, fps: float) -> float`: RMS over joints and frames of the third time-difference (jerk), in units/s³.
  - `anchor(j_m: np.ndarray, up: np.ndarray, height_ft: float, throws: str, release_index: int) -> np.ndarray` returns Statcast ft. It rotates so up is +z and the pelvis's forward direction at release, made horizontal, points to −y (toward the plate). It scales so the standing height (the median head-to-lower-foot distance over the first 0.2 s, ×1.07 for the top of the head; the factor gets a why-comment) equals `height_ft`. It translates so the pivot foot (the throwing side) at frame 0 is at x = 0, y = 60.5 ft, z = 10/12 ft. It also locks the feet: while a foot's speed is below a threshold, its horizontal position is held, so there's no sliding.
  - `release_values(j_ft: np.ndarray, release_index: int, throws: str) -> dict` with `release_x_ft`, `release_z_ft`, `extension_ft` (60.5 − hand y) and `arm_angle_deg` (the throwing shoulder → hand line above horizontal).
  - `correct_to_mean(j_ft, release_index, foot_strike_index, throws, target: dict) -> np.ndarray` (target has `release_values`' keys) is a one-time whole-body correction. It translates the whole sequence by the x and z gap, and scales the stride (the horizontal y displacement of all joints relative to the pivot foot at frame 0) so the extension matches. If the arm-angle gap is more than 2°, it rotates the throwing arm chain about the shoulder by that gap through the arm-action phase (foot_strike → release), easing in. The result's `release_values` match the target to 0.05 ft and 1°.

- [ ] **Step 1: Write the failing tests.** All inputs are synthetic and labelled so.
  - `test_average_aligns_release`: two sinusoidal "deliveries" of different tempo, with release at different frames. The returned release_index matches `pre_s × fps_out`, and the averaged throwing hand peaks in speed at release_index ±1.
  - `test_average_drops_short_clip`: a clip with only 0.3 s before release is excluded, and the reason is reported.
  - `test_smoothing_reduces_jitter`: a smooth curve plus seeded noise (`rng(0)`). `jitter(average)` is less than half of `jitter(raw)`.
  - `test_anchor_height_and_rubber`: a synthetic standing figure 1.8 m tall, rotated arbitrarily, with `up` given. After `anchor(height_ft=6.25)`, the standing height is 6.25 ± 0.05 ft, the pivot foot at frame 0 is at (0, 60.5, 10/12) ± 0.01, and the figure faces −y.
  - `test_feet_do_not_slide`: a planted foot with 1 cm of synthetic drift has a horizontal range below 0.01 ft while planted.
  - `test_release_values`: a hand straight above the shoulder gives 90°; level with it, 0°; hand y = 54.0 gives extension 6.5.
  - `test_correct_to_mean`: after correction, `release_values` match a target 0.5 ft and 4° away, within the stated tolerances.
- [ ] **Step 2:** Run them; FAIL. Implement; PASS.
- [ ] **Step 3:** Commit.

### Task 5: Build script, metrics, export and heights

**Files:**
- Create: `scripts/m5_build.py`
- Modify: `src/pitchsim/data.py` (add `fetch_heights(ids) -> pd.Series` from `https://statsapi.mlb.com/api/v1/people/{id}`, parsing `height` like `6' 3"` to inches, cached at `data/raw/people.csv`)
- Modify: `src/pitchsim/export.py` (`pitcher.height_in` from `fetch_heights`; the schema stays 2, since an added field is backward compatible) and `scripts/export_viewer.py` if needed
- Test: `tests/test_data.py` (height parsing), `tests/test_export.py` (`height_in` present when given)

**Interfaces:**
- Consumes: Tasks 1–4.
- Produces:
  - `web/public/data/deliveries/<slug>.json` (gitignored with `web/public/data/`) containing `{"schema": 1, "pitcher_id", "slug", "throws", "height_in", "fps", "release_index", "pre_s", "post_s", "joints_ft": [[[x, y, z] × 24] × T], "n_clips", "lifter"}`. The joints are after anchoring and `correct_to_mean`, in Statcast ft, SMPL order.
  - `web/public/data/deliveries/<slug>.clips.json`: `{clip_name: {"release_frame", "fps", "release": [x, y, z], "arm_angle_deg", "split"}}` for every screened-ok clip. This is for the eye-test page; numbers only.
  - `results/m5/metrics.csv` and a printed table.

- [ ] **Step 1: Write the failing tests.**
  - `test_parse_height`: `6' 3"` gives 75 and `5' 11"` gives 71.
  - `test_export_height`: `export_pitcher(..., height_in=75)` writes `pitcher.height_in == 75`.

  Run them; FAIL. Implement; PASS.
- [ ] **Step 2: Implement `m5_build.py <clips dir>`.**
  1. Load the train clips' lifts and release frames, compute landmarks, then `average_delivery` (`pre_s` = the largest value at least 85% of train clips cover, capped at 2.5 s), then `anchor`.
  2. **Metrics before correction** on test clips: anchor each test clip's own lift the same way, and compute `release_values` against its Statcast row. Report median and 90th-percentile errors with game-bootstrap 95% intervals (resample held-out games, 1000 draws, seed 0) for release x and z, extension and arm angle.
  3. **The averaging cost:** compare the averaged delivery's `release_values` with each test clip's Statcast values, using the same errors.
  4. **Jitter:** the median per-clip raw jitter against the final delivery's.
  5. **Yield table:** downloaded, dropped by reason, used for train and test.
  6. **Release-frame error** from the Task 2 labels.
  7. Apply `correct_to_mean` toward the **season mean** of his 2026 bases-empty Statcast release values, then write both JSON files.
- [ ] **Step 3:** Re-run `scripts/export_viewer.py` over the six pitchers' CSVs so every pitcher JSON carries `height_in`. Run the whole pytest suite: PASS.
- [ ] **Step 4:** Commit the code and tests (no data or results CSVs).

### Task 6: Viewer timeline starts before release

**Files:**
- Modify: `web/src/state.js`, `web/src/data.js`, `web/src/main.js` (loading, `?t=`, cameras)
- Test: `web/test/state.test.js`

**Interfaces:**
- Consumes: the delivery JSON (may be missing).
- Produces:
  - `pickPitcher` takes an optional `delivery` (object or null) and stores it in state.
  - `preroll(s) -> number` gives `s.delivery?.pre_s ?? 0`.
  - `goPitch`, `nextPitch`, `prevPitch` and `replay` start at `t = -preroll(s)`.
  - `seek` clamps to `[-preroll(s), flightEnd(s)]`.
  - `tick` advances from negative t at `dt × speed`, so the windup slows with the speed setting (Ryan's choice).
  - `landed()` stays false for t ≤ 0 on an unseen pitch (it already requires `t > 0`).
  - `loadDelivery(slug) -> Promise<object|null>` in data.js returns null on a 404 without logging an error.

- [ ] **Step 1: Write the failing tests** (the existing tests must stay unchanged and green).
  - `preroll: replay starts at -pre_s with a delivery, at 0 without`.
  - `seek clamps to -pre_s`, and to 0 for a pitcher without a delivery (`?t=-1` gives 0).
  - `tick from -pre_s reaches release after pre_s / speed seconds`: at speed 0.25, 4 × pre_s of dt reaches t = 0 ± 1e-9.
  - `no spoiler during the windup`: at t = −0.5 on an unseen pitch, `landed` is false and `seen` is unchanged. A setMode during the windup doesn't reveal the result.
- [ ] **Step 2:** Run `cd web; node --test test/*.test.js`; FAIL. Implement; PASS.
- [ ] **Step 3: In main.js:**
  - load the delivery alongside the pitcher JSON;
  - before release (t < 0), ride-along and the trails use `max(t, 0)`;
  - the ball, ghost, trail and no-spin line are hidden before release (the ball is handled in Task 7).
- [ ] **Step 4:** Commit.

### Task 7: The figure throws

**Files:**
- Modify: `web/src/pitcher.js` (and `web/src/skeleton.js` only if the scaling needs it), `web/src/main.js`, `web/src/ball.js` if needed
- Test: `web/test/pitcher.test.js`

**Interfaces:**
- Consumes: the delivery JSON; `pitch.release`, `pitch.arm_angle_deg`; `pitcher.height_in`.
- Produces:
  - `makePitcher(throws, heightIn = 75)` builds a figure whose standing height scales with `heightIn / 75`.
  - `poseAtTime(figure, delivery, t, release, armAngleDeg)`:
    - poses from the delivery frame at time t, interpolating linearly between frames, with `toScene` conversion;
    - blends the throwing arm toward the existing two-bone IK solution that puts the hand on `release`, with weight `w(t)` = smoothstep from 0 at t = −0.15 s to 1 at t = 0, and back to 0 by t = +0.12 s.
  - `handPosition(figure, throws) -> THREE.Vector3`: the throwing knuckles' world position, so main.js can draw the ball in the hand before release.
  - Pitchers without a delivery keep `poseAtRelease` exactly as now.

- [ ] **Step 1: Write the failing tests.** Use a small synthetic delivery fixture, `web/test/fixtures/delivery-synthetic.json`, labelled synthetic: 24 joints, a simple arm swing, release_index 60 at 60 fps.
  - `hand reaches the release point at t = 0`: within 0.01 ft of `toScene(release)`.
  - `far release is reached without a snap`: with the release 0.8 ft from the delivery's own hand, the hand moves at most 0.25 ft between consecutive 1/240 s samples from t = −0.3 to t = 0.1 (a snap would be a single large jump), and the elbow angle stays below 178°.
  - `before the blend the delivery is untouched`: at t = −0.5, the hand equals the fixture frame (within 1e-6).
  - `height scales the figure`: the head-top height for 70 in and 80 in is height/12 ± 0.1 ft, and the hand still reaches the release.
- [ ] **Step 2:** Run; FAIL. Implement; PASS.
- [ ] **Step 3: Wire it in main.js.**
  - With a delivery, `poseAtTime` runs every frame.
  - Before t = 0 the ball is drawn at `handPosition`; from t = 0 it follows the flight.
  - `?view=side&t=-0.8` gives a still for screenshots.
- [ ] **Step 4: LOOK REVIEW, STOP for Ryan.**
  - Save screenshots at t = −1.2, −0.6, −0.2, 0 and 0.2 (side and umpire views) to `results/m5/screens/`.
  - Give him the localhost link. He checks for smoothness, foot sliding and the hand arriving smoothly.
  - Fix what he flags before continuing.
- [ ] **Step 5:** Commit.

### Task 8: Eye-test page

**Files:**
- Create: `web/compare.html`, `web/src/compare.js`
- Modify: `web/vite.config.js` (add compare.html as a second entry if needed)

**Interfaces:**
- Consumes:
  - `deliveries/<slug>.json` and `<slug>.clips.json`;
  - `makePitcher`, `poseAtTime` and `handPosition` from Task 7;
  - the park build from `web/src/park/index.js` (or a plain ground if that is heavy).
- Produces: a page where Ryan picks a local mp4 with `<input type="file">`. The clip is never fetched from the server.
  - The name `<game_pk>_<ab>_<pitch>.mp4` is looked up in clips.json to get its release frame, fps and that pitch's release.
  - The video on the left and the replica on the right play in sync: replica t = `video.currentTime − release_frame / fps`.
  - The replica's hand is pinned to that clip's release.
  - The camera is behind and above the mound, like the center-field broadcast (my call; a fixed preset is fine).
  - Controls: play/pause, speeds 1, 0.25 and 0.1, step one frame with ← and →. It shows whether the clip was train or test.
  - An unknown filename shows "clip not in this pitcher's index" and does nothing else.

- [ ] **Step 1:** Build the page. Use pure helpers in `compare.js` (`clipKey(filename)`, `replicaTime(videoTime, releaseFrame, fps)`) with tests in `web/test/compare.test.js`:
  - `clipKey("824442_12_3.mp4")` returns `"824442_12_3.mp4"`, and a non-matching name gives null;
  - `replicaTime(1.0, 30, 30)` returns 0.
- [ ] **Step 2:** Run the web tests; PASS. Check it in the browser with a real clip from `data/video/`.
- [ ] **Step 3: EYE TEST, STOP for Ryan.** Give him the link and the clip folder path. He compares 3–5 clips, including test clips.
- [ ] **Step 4:** Commit.

### Task 9: Results and wrap-up

**Files:**
- Create: `results/M5.md`
- Modify: `README.md` (an M5 section: the commands in order, the `.venv-pose` setup, the SMPL note, and the warning that clips are personal research only)

- [ ] **Step 1: Write `results/M5.md`:**
  - the spike outcome;
  - the yield table;
  - the release-frame error;
  - held-out release x, z, extension and arm-angle errors with intervals, before correction;
  - the averaging cost;
  - jitter, raw against final;
  - Ryan's eye-test verdict in his words;
  - what's still wrong. Expect extension to be the weakest number, from the depth limits of a front-on camera.

  State plainly what M5's "done when" still needs: more pitchers. The pipeline takes a pitcher, so the follow-up is to run it.
- [ ] **Step 2:** Run the full pytest and web suites; both PASS. Commit.
