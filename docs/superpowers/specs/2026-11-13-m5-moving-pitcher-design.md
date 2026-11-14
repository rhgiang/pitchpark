# M5: The moving pitcher (design)

Date: 2026-11-13. Branch: `m5-moving-pitcher`. Project brief: `CLAUDE.md`, section M5.

## Intent

The viewer's pitcher figure currently holds a single pose at ball release. M5 makes it **throw**: the
whole delivery, from the start of the windup through leg lift, stride, arm path, release and
follow-through. The motion is recovered from real video of the pitcher, and the ball leaves the hand
exactly where and when Statcast says it did.

Ryan's priority, in his words: "as long as it is a pretty smooth pitching motion that is relatively
accurate I'd be happy". **Smooth, good-looking motion beats perfect accuracy**, but accuracy is still
measured with numbers, because "the eye test can say one thing, and being backed by numbers is even
better". Even Trajekt doesn't reproduce the pitcher's body today, so a believable delivery is the win.

## Goals

1. One **bases-empty delivery for Shohei Ohtani** (2026 season), recovered from Baseball Savant's
   per-pitch broadcast clips and played on the existing viewer figure.
2. The figure is **scaled to his real height** (MLB StatsAPI `people` endpoint; 6 ft 3 in).
3. **One shared delivery for every pitch type** (Ryan's call). Per pitch, the end of the arm is bent
   (the viewer's existing two-bone IK) so the hand meets that pitch's Statcast release point. The body
   moves the same every time; only the hand's final spot differs, as it does in reality.
4. The ball leaves the hand at the release frame, and the M1 flight starts from there, in sync, at
   every playback speed the viewer offers (1x, 1/4x, 1/10x).
5. **Numbers** on held-out clips (below), **and** an eye-test page that plays a real clip next to the
   replica in sync.

## Non-goals

- The stretch / runners-on delivery (Ryan: "just one to start"). The bases-empty delivery is my call:
  it has the most clips (894 of 1,335 pitches) and the full leg lift.
- Per-pitch-type deliveries or "tells" (Ryan: one delivery).
- Other pitchers in this pass. The pipeline takes a pitcher id and must not hard-code Ohtani, but
  running it for more pitchers is the follow-up that completes CLAUDE.md's "several pitchers"
  criterion.
- A new character model or body-shape fitting. The existing stylised figure (`web/src/pitcher.js`,
  SMPL 24-joint skeleton with Mixamo names, `web/src/skeleton.js`) stays.
- Fielding or anything after the follow-through.

## Anti-goals (would be a failure even if the numbers pass)

- **Jitter or twitching**, especially around release, where broadcast frames blur the arm.
- Feet sliding on the mound ("ice-skating"), or limbs going through the ground or body.
- The hand **snapping** onto the release point at the last frame, instead of arriving there smoothly.
- Committing or redistributing any video. Clips live under `data/` (gitignored) on Ryan's machine only
  (CLAUDE.md: video is for personal research).

## Constraints and facts already checked

- **Clip lookup works** (checked 2026-11-13 on one Ohtani 2026 pitch). Statcast CSVs have no video id,
  but the StatsAPI live feed (`/api/v1.1/game/{game_pk}/feed/live`, already used for weather in
  `pitchsim/data.py`) gives `liveData.plays.allPlays[at_bat_number - 1].playEvents[]` where
  `isPitch` is true. The `pitch_number`-th pitch event carries `playId`, and its pitch type and speed
  matched the Statcast row (FF, 99.5 mph). `https://baseballsavant.mlb.com/sporty-videos?playId=<id>`
  returns a page containing one direct `https://sporty-clips.mlb.com/....mp4` link. Each clip is one
  pitch, mostly the center-field broadcast camera.
- **Hardware:** NVIDIA RTX 5060 Ti (16 GB), 32 GB RAM, Windows 11. This is a new (Blackwell) card:
  PyTorch needs a CUDA 12.8+ build, and some research repos pin older versions that won't run on it.
- **Dependencies:** CLAUDE.md requires asking before adding PyTorch. **PyTorch is approved by Ryan for the 3D
  step (2026-11-13)**, as a CUDA 12.8+ build that runs on the RTX 5060 Ti. Heavy video/pose dependencies go in a **separate environment**
  (e.g. `.venv-pose`) or an optional extra, so `pip install -e .` and the existing 89 tests stay light.
  Lightweight 2D pose (e.g. RTMPose via `rtmlib`/ONNX, or MediaPipe) is preferred where it is enough.
- **Body model weights:** if the chosen 3D model needs the SMPL body-model files, those need a free
  registration on the SMPL site (non-commercial research licence), done by Ryan. The plan must say
  exactly when and what he downloads.
- **Camera geometry:** center field is a front-on, telephoto view. Sideways position and heights are
  well observed; **depth (stride length, extension) is the weak axis**. The numbers must show that
  honestly rather than hide it.
- Coordinates and units follow the project (`feet`, Statcast x/y/z; viewer `toScene`).
- Reproducible from scripts in `scripts/`, fixed seeds, `results/M5.md` at the end.

## The pipeline (decided shape; tools inside each step are the builder's)

A new package `src/pitchsim/pose/`, designed cleanly enough to reuse (CLAUDE.md: the same skill as
teaching robots from human video).

1. **Clips.** Pitch rows → `playId` → mp4, cached under `data/video/<pitcher>/<game_pk>_<ab>_<pitch>.mp4`.
   Never re-download what's cached. Bases-empty means `on_1b`, `on_2b`, `on_3b` all empty.
2. **2D pose per frame**, with per-joint confidence.
3. **Clip screening.** Drop clips that aren't the usual center-field angle, have a camera cut or
   replay inside the delivery, or lose the pitcher. Report how many survived and why each was dropped.
4. **Release frame per clip.** Needed for alignment, sync and scoring. How to find it is the builder's
   call (e.g. peak throwing-hand speed in 2D), but its accuracy must be checked by hand on a handful of
   clips and reported in frames.
5. **3D lift.** A monocular human-body model gives a 3D pose sequence per clip. Pick it with a quick
   compatibility spike on the RTX 5060 Ti. Candidates in my order of preference: a world-grounded
   video model (GVHMR or WHAM), then a per-frame mesh model (4D-Humans / HMR 2.0). Use the first one
   that installs and runs. The spike result goes in `results/M5.md`.
6. **One delivery.** Line up the training clips' deliveries on their release frames, time-normalise the
   phases before and after release, average, and smooth. This average is what makes it smooth. Its
   timing is real time from the clips' frame rate.
7. **Anchor.** Scale to his height, put the pivot foot on the rubber, face home plate, and keep the feet
   planted when they are planted (no sliding).
8. **Export** the delivery as JSON for the viewer (joint rotations or positions per frame in the
   existing 24-joint order, the release frame index, fps). No video or video-derived images go in it.
9. **Viewer playback.** The figure plays the delivery before each pitch, the throwing hand eases onto
   that pitch's Statcast release point over the last part of the arm action (no snap), the ball appears
   in hand and starts its M1 flight at the release frame, then the follow-through plays.

## Decisions from the review round

- **The windup is part of the pitch timeline (Ryan).** Time runs from the start of the delivery
  (t < 0) through release (t = 0) to the end of the flight. The speed setting slows the windup too, and
  it can be scrubbed. Release stays t = 0, so the M1 flight, `?t=`, and the existing tests that start at
  release keep their meaning. Whatever "play from release" paths exist (next/previous pitch, the pitch
  strip, replay) should start from the start of the delivery unless a URL `t` says otherwise. Cameras
  that follow the ball (ride-along) wait at release until t = 0. These are the builder's details.
- **Match his average release once (my call).** After anchoring, the averaged delivery gets one
  whole-body correction (shift, plus stride scale if needed) so its release lands on Ohtani's **mean**
  Statcast release height, side, extension and arm angle for the season. Per pitch, the arm IK then only
  covers the small pitch-to-pitch differences, so no locked elbow or snap. The held-out numbers are
  measured **before** this correction.
- **Other pitchers keep the static release pose (my call)** until their own delivery is recovered.
  Playing Ohtani's motion on them would pass his delivery off as theirs. **Every pitcher is scaled to
  his StatsAPI height**, which is cheap and true.
- **Release-frame labels (my call):** the builder labels the release frame by hand on about 20 clips
  from extracted frames, spread across games and pitch types. Ryan can spot-check them. The detector's
  frame error is reported on those.
- **Eye-test page (my call):** side by side only (no overlay on the video, so no per-clip camera needs
  exporting). The replica's hand is pinned to that clip's own Statcast release, as in the viewer.

## How it's judged

The split is **by game**, as in every milestone: clips from held-out games are never used to build the
delivery. All numbers are measured **before** the hand is pinned to Statcast, because pinning would make
them perfect by construction.

| Check | Measured on | Reported |
|---|---|---|
| Release height and side vs Statcast `release_pos_z/x` | held-out clips, recovered hand at the release frame after anchoring | median and 90th-percentile error, inches |
| Extension vs Statcast `release_extension` | same | median error, inches (expected to be the weakest number) |
| Arm angle vs Statcast `arm_angle` | same | median absolute error, degrees |
| The averaged delivery vs held-out clips | held-out clips' release values vs the averaged delivery's | the same three errors, so we see what averaging costs |
| Smoothness | raw per-clip 3D vs the final delivery | a jitter number (e.g. RMS joint jerk); final must be clearly below raw |
| Release-frame detection | a handful of hand-labelled clips | error in frames |
| Clip yield | all bases-empty clips | downloaded, screened out (by reason), used |

No pass threshold is set in advance for the accuracy numbers. **Done when** they are reported with
their intervals, the jitter number shows the final delivery is smooth, and the delivery plays in sync
with the simulated pitch in the viewer. Ryan's eye test on the comparison page is the final sign-off.

**Eye-test page:** plays a real clip and the replica side by side, synced on the release frame, with
slow motion and frame stepping. The clip is loaded from Ryan's disk through a file picker (or a
dev-only local route), so no video is ever served from `web/public` or committed.

## Left to the builder

The 2D pose tool, the 3D model (by the spike), release-frame detection method, screening rules,
alignment and smoothing method, the delivery's JSON layout, how many clips to use, and the eye-test
page's layout. Each choice gets a short "why" comment, per CLAUDE.md.
