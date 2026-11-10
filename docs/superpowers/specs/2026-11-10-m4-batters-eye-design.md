# M4: the batter's eye (a stylised ballpark you can watch real pitches in)

Date: 2026-11-10. Project brief: `CLAUDE.md`. Builds on M1 (flight), M2 (generated pitches) and M3
(learned push, `results/M3.md`).

## Intent

Until now every pitch has lived in CSVs and plots. M4 is the first time Ryan can *watch* one:
stand in the batter's box, see the pitcher on the mound, and watch a real at-bat come at him at
real speed, then flip any pitch between what Statcast recorded and what our model flies.

Two reasons it matters:
- **It's the display layer for everything after it.** M5 puts a moving pitcher here, M7 plays
  simulated at-bats here. M4 builds the stage and the data path; later milestones only change
  what's fed in.
- **World building is the part Ryan most wants to do well (Ryan).** He's drawn to Three.js
  worlds like Messenger (messenger.abeto.co) and a stylised cricket stadium on a floating
  island. The viewer should feel like a place, not a debug plot.

Ryan is building this to learn. Code explains the *why* (camera geometry, timing, shaders) in
short comments. Small steps with Ryan's OK between them.

## Goals

### The look (Ryan)
- **Stylised and cartoon-like**, not realistic: closer to Messenger and the cricket stadium than
  to a photoreal scene. Flat colour, soft shading, outlines. *(Claude's call: toon materials plus
  an outline pass; no pixel-art blocks.)*
- **A ballpark full of small details**, in the cricket stadium's spirit: stands, scoreboard,
  light towers, foul poles, bullpen, things beyond the outfield.
- **Time of day:** day, sunset and night under the lights, switchable. *(Claude's call, from the
  cricket reference; Ryan didn't object.)*
- **A pitcher that reads clearly (Ryan):** more refined than the cricket figures, because the
  point is watching him pitch.

- **Quality bar (Ryan, 2026-11-10): "as close and as good as what we've seen so far."** The
  references are Messenger (messenger.abeto.co) and the cricket-stadium clip
  (x.com/vib3coded/status/2117584501744312627). The starter park should be judged against them
  side by side, not against "good enough for a viewer". This bar applies to this first,
  coded iteration, not a later pass. Polish (lighting, colour, detail
  density, camera moves) is real scope, not a nice-to-have.
- **What makes those references work** (from frame-by-frame viewing, 2026-11-10), as targets:
  one tight palette, everything outlined and flat-shaded, soft baked-looking shading and contact
  shadows, dense small props, a wide camera that can sweep from the whole world down to ground
  level, and time-of-day changes that recolour the whole scene. Messenger's code shows custom
  toon shaders, outlines, lightmaps/AO maps, instancing and SMAA; the coded park should approach
  that look with code-generated equivalents (e.g. vertex-colour or generated AO, instanced props).

### Who builds the park (Ryan)
- **Claude generates a starter ballpark; Ryan builds his own afterwards.** The park must be
  swappable without touching the rest of the viewer.
- **Built entirely in code for this version (Ryan).** Ryan has used a little Blender and will
  learn more; no Blender assets in M4. A Blender park or pitcher can replace the coded one later.

### The views (Ryan)
- **A. Batter's eye (main):** camera at a batter's eye height in the box, aimed at the strike
  zone, strike zone in frame, pitcher on the mound. No batter in the scene. Ball at real speed.
- **Side view:** whole flight, release to plate, to see drop and arc.
- **Free camera:** orbit and zoom anywhere, including pulled right out to the whole park.
- **Ride with the ball:** the camera chases the ball, with its spin shown.
- Out (Ryan's picks): catcher/umpire view, centre-field TV view (better once M5 adds a delivery).
- *(Claude's call)* Switch views by key or button; any pitch can be replayed in any view.

### What happens (Ryan agreed to Claude's proposal)
1. **Pick a pitcher** (the five modelled ones; nothing hardcoded to five).
2. **Pick an at-bat** from his **held-out games**, or "random at-bat". *(Claude's call: held-out
   games only, so the simulated pitches you watch were never fitted on those games.)*
3. **It plays out:** each pitch in the real order, with live count (balls, strikes, outs,
   inning), pitch type, speed, and the real result (`description`, plus `events` when the
   at-bat ends). No batter, so results come straight from Statcast.
4. **Real vs simulated:** one toggle flies the pitch as Statcast's path or as our model's
   (M1 + M3 from the release point), plus a **ghost mode** showing both at once.
5. **Replay any pitch, slow motion, switch view.**
6. **Tunneling read-outs:** per pitch, vertical and horizontal approach angle and perceived
   velocity; between consecutive pitches, separation at the decision point.

M7 replaces step 3's source (M6 picks the pitch, M2 generates it); the viewer stays the same.
Fully simulated at-bats are **not** M4.

## Honesty about what's "real"

- "Real" in the viewer is **Statcast's 9-parameter constant-acceleration fit**, evaluated from
  release to the plate, not raw tracking points. The UI labels it "Statcast fit".
- "Simulated" is M1 + M3 (per-pitch push), started from `release_state`, with that game's air
  density. `results/M3.md` documents the known gaps (release-state artefact, misses of about
  1.5–3 in). The ghost mode is there to show them, not hide them.
- **Pitches with no model (Ryan, 2026-11-10):** pitch types M3 skipped (too few training
  pitches, e.g. Ohtani's splitter, Misiorowski's changeup and sinker) play the Statcast path only,
  labelled "no model for this pitch type", with the real/simulated toggle and ghost mode
  disabled for that pitch. Rows with no tracking at all still advance the count, show "no
  tracking", and have no flight. At-bats are never dropped for either reason.
- Nothing synthetic appears in the real-pitcher viewer. Synthetic data, if used for tests, is
  labelled and kept out of `web/` data.

## Architecture

- **Python is the source of truth.** `src/pitchsim/export.py` writes JSON; the browser only plays
  it back. No physics in JavaScript. *(Claude's call)*
- **Export (per pitcher):** held-out at-bats, each pitch with: count before the pitch, inning,
  outs, batter side (`stand`), `sz_top`/`sz_bot`, pitch type and name, speed, spin rate and
  axis, `arm_angle`, release point and extension, result, and two sampled trajectories (Statcast
  fit and simulated) as time-stamped x/y/z points from release to the plate, plus tunneling
  numbers. Units and coordinates in the file header: feet, seconds, Statcast axes (x toward the
  catcher's right, y from plate toward pitcher, z up).
- **`src/pitchsim/tunneling.py`** (brief): approach angles at the plate, perceived velocity from
  extension, pair separation at the decision point.
- **`web/`:** Three.js + Vite, plain JavaScript. *(Claude's call: no React or TypeScript.)*
  Structure the scene so the park, the pitcher figure, and the data are separate modules, so
  Ryan's park and M5's pitcher drop in.
- **The pitcher figure:** built in code with a **real skeleton** (named joints for spine,
  shoulders, elbows, wrists, hips, knees, ankles) so M5 can drive it and a Blender character
  can replace it on the same skeleton. *(Claude's call)* A Three.js `SkinnedMesh` with `Bone`s,
  in SMPL's 24-joint layout (what M5's 4D-Humans/WHAM output) with Mixamo-style bone names (what
  Blender rigs usually use), T-pose rest pose; the mapping between the two naming schemes is
  written down once, next to the skeleton. In M4 it holds a static pose near
  release; the ball appears at Statcast's release point, with the throwing hand placed there
  and the arm set at `arm_angle`. Handedness from `p_throws`.
- **Strike zone:** 17 in wide at the plate, top and bottom from that pitch's `sz_top`/`sz_bot`.
- **Timing:** real time by default (about 0.4 s release to plate), plus slow motion.
- **Batter side:** the camera stands in the box the real batter used (`stand`).

## Measurable checks (CLAUDE.md: every milestone gets a number)

1. **The viewer shows what Python computed.** An automated test plays the exported data through
   the viewer's own trajectory code and checks the ball crosses the plate (y = 17/12 ft) at the
   exported position and time, within a small stated tolerance. *(Claude's call: Node's built-in
   test runner, `node --test`, so no extra test dependency.)*
2. **Perceived velocity vs Statcast.** Our perceived velocity is compared with Statcast's own
   `effective_speed` on held-out pitches; report median absolute difference (mph) per pitcher.
3. **Approach angles.** Vertical approach angle checked against a public reference (Baseball
   Savant) if one is published for these pitchers; otherwise against hand-worked cases from the
   9-parameter fit, and the write-up says which.
4. **Tunneling math:** unit tests on hand-built trajectories (two identical pitches have zero
   separation; a known offset gives a known separation).

**Done when:** Ryan can pick a pitcher, pick an at-bat, and watch it in the batter's-eye view
with real vs simulated side by side (ghost mode), switch to the three other views, and checks
1–4 are reported in `results/M4.md`.

## Constraints

- Three.js and Vite are approved by the brief. Any other npm dependency: ask Ryan first.
- Pinned versions; `web/` builds and runs with `npm install && npm run dev`.
- Exported JSON for real pitchers lives under `results/` or `web/public/data/` and is gitignored
  (it is derived from `data/`, which is gitignored). The repo must regenerate it from a script
  in `scripts/`.
- Runs smoothly on Ryan's desktop (NVIDIA GPU) at 60 fps. Phones are a non-goal.
- Commits carry **no `Co-Authored-By` trailer** and no Claude attribution (Ryan's standing rule).
- Statcast data only; no video (M5).

## Non-goals

- Fully simulated at-bats, pitch selection, generated pitches in the viewer (M6/M7).
- The pitcher's delivery motion (M5). M4's figure doesn't animate.
- Blender assets, Ryan's own park (after M4), VR (stretch), sound, mobile.
- A hitter, swings, or batted-ball flight.
- More training data (Ryan, 2026-11-10: after M4). Ohtani 2026 and Misiorowski 2025 would let
  more pitch types be modelled. That is its own step after M4, with a season-mixing rule. The
  viewer must pick up new models with no code change.

## Anti-goals

- Physics in JavaScript that could drift from the tested Python.
- A pretty view that's wrong: wrong distances, wrong timing, a zone in the wrong place, or the
  "real" path passed off as tracking when it's the fit.
- Showing simulated pitches as if they were real, or unlabelled.
- A park that's welded to the rest of the code so Ryan can't replace it.

## Left to the builder

- Exact eye height and stance position in the box (state the numbers and their source).
- Outline technique, toon ramp, colours, park layout and props, within the look above.
- Trajectory sample spacing and JSON layout; the decision-point distance within the brief's
  23–25 ft, stated.
- How spin is shown in ride-with-the-ball (real spin is ~40 rev/s, faster than a frame rate
  can show; say what's done).
- UI layout and controls.
