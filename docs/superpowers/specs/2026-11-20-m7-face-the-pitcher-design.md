# M7: Face the pitcher — design

Date: 2026-11-20. Status: draft for Ryan's review.

## Intent

A virtual Trajekt. You pick one of the 8 pitchers and a batter side, and he throws you a live,
newly generated at-bat: a sequence of pitches in and out of the zone, each chosen the way he really
chooses (M6), generated like his real pitches (M2), flown by our physics (M1 + M3), thrown by his
recovered delivery (M5), and seen from the batter's box (M4). Nobody hits. It is the milestone that
joins M1–M6 into one experience, and it is the résumé centrepiece of the hosted site
(pitchpark.vercel.app).

Why it matters (Ryan): it should feel like stepping in against him on a Trajekt. The "why he threw
it" numbers make the sequencing model visible, which is the most impressive part to show.

## Goals

1. **Two modes in the viewer.**
   - **Replay** is the current viewer, unchanged: real Statcast at-bats re-flown by our physics.
   - **Face him** is new: generated at-bats. Every pitch is simulated and labelled "Simulated".
2. **All 8 viewer pitchers can be faced.** Ohtani, Misiorowski and Sale have M2/M6 models now. Cease,
   King, Sánchez, Schlittler and Yamamoto get M2 and M6 fitted and graded as part of M7, with the
   same checks M2 and M6 used, written into results/M2.md and results/M6.md.
3. **Batter side.** You stand in as a lefty or a righty. The camera goes to that box and every model
   is conditioned on that side.
4. **Batter switch (Ryan: "a mode for both").**
   - **None (called only):** in the zone is a called strike, outside is a ball. Each at-bat ends in
     a walk or a strikeout.
   - **Average hitter:** an invisible typical batter decides take / swing-and-miss / foul / in play,
     learned from that pitcher's real Statcast pitch results against that batter side. At-bats end
     in a strikeout, a walk or "ball in play". The site never invents a hit or an out.
5. **"Call the pitches" switch, off by default (Ryan).** When on, you call each pitch while it is
   still in flight. The call is ball/strike, plus an optional pitch-type pick. A running score is
   shown ("7/10 calls, 4/6 types"). When off, pitches come at a natural pace like a Trajekt session.
6. **Reveal card after each pitch (Ryan).**
   - Placement and look: off to the side, never over the zone or the pitcher, and subtle, with a
     translucent background in the style of the existing bottom-right panel.
   - Contents: pitch type, speed, a mini zone with where it crossed, its break, the result (called
     strike, ball, swinging strike, foul, in play) and **why he threw it**: the model's probability
     for that type in this count and side (e.g. "1-2 vs RHB: 41% SL").
   - Behaviour: it fades in for a few seconds without pausing play, and adds the pitch to a running
     at-bat strip like a broadcast pitch tracker.
7. **Measured checks.** These are held-out and per pitcher, and none of the five new pitchers ships
   without them:
   - **M2 and M6 for the five new pitchers:** the M2 classifier test with the floor-relative rule;
     M6 log loss beats the count baseline on held-out games (5 game folds); calibration reported.
   - **Simulated at-bats vs real at-bats on held-out games**, Average-hitter mode:
     - pitch mix by count;
     - zone rate;
     - first-pitch strike rate;
     - pitches per at-bat;
     - outcome rates (K / BB / in play).
     Each with an interval or test so "matches" is a number, not a feeling.
   - **Average hitter:** held-out log loss of its take / whiff / foul / in-play prediction against a
     count-only baseline.
   - **Re-aim error:** how far the browser's re-aimed flight lands from the sampled crossing point,
     against a true Python re-simulation (see "How", step 4).
   - Results go in `results/M7.md`.

## Non-goals

- No hitting, no bat, no batted-ball flight, no hit/out outcomes.
- No game state beyond the count: no outs, runners, score or innings. These were noise in M6.
- No server: the site stays static on Vercel.
- No changes to Replay mode's behaviour or look.
- No real-time model training in the browser.

## Anti-goals (a failure even if it works)

- A generated pitch shown without the "Simulated" label, or mixed into the real-pitch data
  (CLAUDE.md: never pass off synthetic data as real).
- Always throwing the most likely pitch. Pitch type and location are always **sampled**.
- A popup or guess UI that covers the zone or the pitcher, or that stalls play when guessing is off.
- A pitcher who can be faced without his checks being reported.
- Breaking Ohtani's signed-off delivery (web/src/signed_off/, frozen): generated pitches go through
  the same pitcher API as real ones.

## Constraints

- Python stays the source of truth for physics and models. The browser only samples from exported
  parameters and plays precomputed flights. JS reproductions of Python sampling get a parity test
  against Python on fixed seeds.
- Static hosting: all M7 data is JSON under web/public/data/ and is committed. Keep each pitcher's
  added data around 1 MB gzipped or less.
- Real Statcast data only for fitting. Fixed seeds. Every number reproducible from a script in
  scripts/.
- No new heavy dependencies; numpy, scipy and scikit-learn are already in. The M6 transformer is not
  used here (the default is logistic + times through the order, which M6 found best or tied).
- Video stays out of it, as before.
- Commits: no co-author trailers.

## How (decided)

These are my calls; Ryan can override.

### The pitch pipeline for one pitch

1. **Context.** Count, batter side, the previous pitch in this at-bat, and times through the order,
   fixed at first time through.
2. **Type.** M6 logistic+tto gives probabilities over his arsenal. Sample one. Export the fitted
   coefficients per pitcher to JSON and compute the softmax in JS.
3. **Target.** The M6 location model for that type (a 2D Gaussian whose mean shifts with count,
   side and previous pitch) gives where he aims. Sample a crossing point. Export its coefficients
   and covariance.
4. **Pitch.** Take a pitch of that type from a per-pitcher **bank**: a few hundred M2 samples per
   type, generated offline and flown by M1 + M3 in Python, stored like the viewer's existing "sim"
   flights (Hermite knots).
   - **Aiming.** M2 models the release angles, but where a pitch is aimed is M6's job (M2's own
     docstring says so). So the bank pitch is re-aimed at the sampled target.
   - **Default re-aim method:** a small rigid aim correction applied in the browser, i.e. a release
     direction change approximated as an offset that grows linearly from 0 at release to the
     required plate offset.
   - **The check that decides it:** Python measures the error of that approximation against a true
     re-simulation with solved release angles, over the bank. If the max error is under about
     0.5 in at the plate, the approximation ships. If not, Python solves the angles per bank pitch
     for a grid of targets, or another exact method the builder picks.
5. **Result.**
   - **None mode:** strike if the crossing point is in the rule-book zone, otherwise ball. Use the
     zone the viewer already draws (the batter's sz_top / sz_bot medians for this pitcher's data).
   - **Average hitter mode:** sample take / swing-and-miss / foul / in play from a small model
     fitted on the pitcher's real pitches against that side.
     - **Features:** count, pitch type, and location (in/out of zone plus distance from the zone
       edge).
     - **Model:** multinomial logistic, smoothed toward a count-only table. Exported as JSON.
     - **Takes:** in the zone is a called strike, outside is a ball (the same zone rule).
     - **Fouls:** a foul with two strikes leaves the count unchanged.
6. **Count update** and end of at-bat: K, BB, or in play (Average hitter only). Then a new at-bat
   starts on a tap, or automatically after a short pause when the user hasn't touched anything.

### UI

- The mode choice sits next to the pitcher picker on the start screen, and as a switch in the main
  controls: **Replay / Face him**.
- Face him controls:
  - batter side L/R;
  - Batter: None / Average hitter;
  - Call the pitches: on/off;
  - Start count: 0-0 by default, any count selectable;
  - New at-bat;
  - pace (pause between pitches).
- All view modes (umpire, batter, TV, etc.), speeds and day/night keep working in Face him mode.
- Ryan sees mockups of the Face him controls, the reveal card and the guess UI before they are built
  (he wants previews for visual changes).
- Guessing works with a keyboard and on a phone. The reveal card and controls fit at phone width.

## Rulings from the builder check (my calls)

1. **The sampled point is the final crossing, not an aim point.**
   - M6's location model is fitted to where real pitches actually crossed, so its spread already
     includes his command scatter.
   - The bank pitch's own release angles are discarded, and it is re-aimed to land exactly there.
     No second scatter is added on top, so zone and ball rates are not double-counted.
   - M2 still supplies release point, extension, speed and spin, which together make the look and
     the break.
2. **Re-aim size.**
   - Offsets of 1–2 ft at the plate, about 1–2° of release angle, are expected and must be covered
     by the check across the full range the location model produces, not just small offsets.
   - If the browser approximation fails the 0.5 in bar, the fallback must stay inside the size
     budget. For example, store per bank pitch the sensitivity of the plate crossing to the two
     release angles and apply that linear correction plus the exact re-flight check. Builder picks.
     Python measures whichever method ships.
3. **Release point vs the delivery.**
   - Generated pitches are pinned exactly like real ones: the viewer already pins the throwing hand
     to each pitch's own release point.
   - Ohtani's frozen code is used through the same pitcher API, unchanged. If his path cannot take a
     per-pitch release, generated Ohtani pitches use his median release instead, and that is noted
     in results/M7.md.
4. **Thin hitter data.**
   - If the average-hitter model does not beat the count-only baseline for a pitcher and side, that
     pair ships with the smoothed count-by-zone table instead.
   - results/M7.md says so. Thin data is never hidden.
5. **Rare cases.**
   - **Pitch types:** follow M2's existing per-type sample minimums. A type too rare to model is
     left out of Face him (renormalise the M6 probabilities) and listed in results/M7.md.
   - **Unseen count/side contexts:** fall back through the models' existing smoothing toward the
     pitcher's mix. No special code.
   - **"Ball in play" endings:** already agreed with Ryan as part of the Average hitter mode.
6. **Parity tests.**
   - **Deterministic pieces, exact match to Python within float tolerance:** type probabilities,
     location mean and covariance, hitter probabilities, and the re-aim offset.
   - **Random sampling, distributional parity:** JS samples on a fixed seed match the target
     distribution (e.g. a chi-square or KS test in the JS test suite). No cross-language RNG
     matching.

## Left to the builder

- Bank size per type within the size budget, and the knot spacing (reuse KNOT_S if the error table
  allows).
- Exact JSON schema for the M7 export (a new `schema` number, versioned like the existing files).
- The exact average-hitter features and smoothing, as long as it beats the count-only baseline on
  held-out games.
- The statistical test used for each at-bat check, as long as each "match" claim has a number.
- How Face him state lives in the URL (deep links like `?face=1&stand=L` would be nice, not
  required).
- Animation timing of the reveal card and the guess window.
