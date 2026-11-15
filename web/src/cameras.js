import { stateAt, duration } from "./trajectory.js";
import { fenceDistFt, FENCE_HEIGHT_FT } from "./park/layout.js";

// Camera placements in Statcast feet (x catcher's right, y plate -> pitcher, z up).

// Eye height: standing eye height is ~0.936 x stature (Drillis & Contini 1966, segment
// proportions); for an MLB-average ~6 ft 2 in hitter that is ~5.8 ft, less ~9 in of batting
// crouch -> 5.0 ft. The crouch amount is our estimate, not a measurement.
export const EYE_HEIGHT_FT = 5.0;
// Batter's box: 4 ft wide x 6 ft long, inner line 6 in from the 17 in plate (Official Baseball
// Rules 2.01 diagram), so it spans |x| = 1.2 to 5.2 ft and y ~ -2.3 to 3.7 ft.
// The two numbers below are camera-framing choices, not a measured stance: |x| = 2.5 ft puts the
// eye over the inner half of the box, y = -2.0 ft keeps it inside the box near the back line.
export const STANCE_X_FT = 2.5;
export const STANCE_Y_FT = -2.0;

// Aim and field of view, also framing choices: picked (by a grid search over every exported
// pitch and over yaw and pitch, 2026-11-11) as the narrowest vertical FOV that, at 16:9 and
// within 95% of the frame, shows for the pitch's own batter side every zone corner, the release
// point, and the end of each visible flight (Statcast fit and simulated): the plate arrival, or,
// for a ball in the dirt, where it reaches the ground (centre at one radius, where ball.js hides
// it). Ryan's rule (2026-11-11): only what the viewer can see has to be in frame; the fit's
// below-ground continuation does not count. Needs 63.1 deg. Checked by test/cameras.test.js.
export const AIM_YAW_DEG = 20; // turned toward the plate (right for a righty, left for a lefty)
export const AIM_PITCH_DEG = -24.5; // looking down
export const BATTER_EYE_VFOV_DEG = 63.5;

/**
 * Batter's-eye camera for batter side `stand` ("R" or "L").
 * A righty stands on the catcher's left (x < 0), a lefty on the right; the aim is mirrored.
 * Use with a perspective camera of vertical FOV BATTER_EYE_VFOV_DEG.
 * @returns {{position: number[], target: number[]}} Statcast feet.
 */
export function batterEye(stand) {
  const side = stand === "L" ? 1 : -1;
  const position = [side * STANCE_X_FT, STANCE_Y_FT, EYE_HEIGHT_FT];
  const yaw = (-side * AIM_YAW_DEG * Math.PI) / 180; // +yaw turns toward +x
  const pitch = (AIM_PITCH_DEG * Math.PI) / 180;
  const dir = [Math.sin(yaw) * Math.cos(pitch), Math.cos(yaw) * Math.cos(pitch), Math.sin(pitch)];
  return { position, target: position.map((v, i) => v + 10 * dir[i]) };
}

// Umpire's view (the main view; Ryan, 2026-11-12): a standing home-plate umpire looking over the
// catcher. Picked by Ryan from a side-by-side with the catcher's eye (70 deg FOV) and 4.5 ft back.
// Depth: the catcher's eyes are at about y = -2.5 ft (he must stay in the catcher's box, which starts
// at the batter's boxes' back line, y ~ -2.3 ft, OBR 5.02(a)); the umpire stands about 2.5 ft behind
// him -> y = -5.0 ft. Height: standing eye height is ~0.936 x stature (Drillis & Contini 1966); for a
// ~6 ft 1 in umpire that is ~5.7 ft, less ~1 ft for the knees-bent "slot" stance he takes to see over
// the catcher's head -> 4.75 ft. Estimates, not measurements.
export const UMPIRE_Y_FT = -5.0;
export const UMPIRE_EYE_HEIGHT_FT = 4.75;
// Aim and FOV: framing choices with the same visible-flight rule as the batter's eye (every exported
// pitch, 16:9, within 95% of the frame: zone corners, release, and the visible end of the
// Statcast-fit, simulated and no-spin paths). 44 deg is the narrowest that passes at this aim
// (controller's grid search, 2026-11-12); checked by test/cameras.test.js.
export const UMPIRE_AIM_PITCH_DEG = -15;
export const UMPIRE_VFOV_DEG = 44;

/**
 * Umpire's-eye camera, the same for both batter sides: on the plate's centre line (x = 0),
 * looking straight out at the pitcher, tilted down so the zone sits mid-frame.
 * @returns {{position: number[], target: number[], vfovDeg: number}} Statcast feet.
 */
export function umpireView() {
  const pitch = (UMPIRE_AIM_PITCH_DEG * Math.PI) / 180;
  const position = [0, UMPIRE_Y_FT, UMPIRE_EYE_HEIGHT_FT];
  return {
    position,
    target: [0, UMPIRE_Y_FT + 10 * Math.cos(pitch), UMPIRE_EYE_HEIGHT_FT + 10 * Math.sin(pitch)],
    vfovDeg: UMPIRE_VFOV_DEG,
  };
}

// Other views. Vertical FOVs are framing choices (the batter's eye needs 63.5 deg; these do not).
export const SIDE_VFOV_DEG = 35;
export const RIDE_VFOV_DEG = 60;
export const FREE_VFOV_DEG = 50;
const RIDE_BEHIND_FT = 6; // camera trails the ball by this much along its flight...
const RIDE_ABOVE_FT = 1.5; // ...and sits this far above it...
const RIDE_SHOULDER_OUT_FT = 1; // ...and, at release, this far out to the throwing side (framing choices)
// Ride aims at the middle of a typical zone at the front of the plate (Statcast sz_bot/sz_top are
// ~1.6/3.4 ft), so the zone stays ahead and centred the whole way in.
const RIDE_AIM = [0, 17 / 12, 2.5];
// Windup follow camera (t < 0 when the pitcher has a delivery), all relative to his pelvis, which
// moves ~5 ft toward the plate over the stride so the camera follows it. Framing choices: 8 ft back
// and 5 ft out to the throwing side shows his back and arm with the plate direction readable; 2 ft
// above the pelvis and aiming at chest height (3.5 ft) keeps head (~6.5 ft), the lifted knee and
// feet in the 60 deg vertical FOV at ~9.5 ft range.
const WINDUP_BEHIND_FT = 8;
const WINDUP_OUT_FT = 5;
const WINDUP_ABOVE_FT = 2;
const WINDUP_AIM_AHEAD_FT = 6; // aim this far toward the plate from the pelvis, so the plate side is in view
const WINDUP_AIM_Z_FT = 3.5;
// Glide into the release viewpoint over this long: slow enough not to jar, quick enough that the
// windup is still mostly followed. Eased (smootherstep) so position and aim start/end at rest.
const WINDUP_BLEND_S = 0.4;
const SIDE_HEIGHT_FT = 5; // side camera height; it aims 1 ft lower (framing choices)

/**
 * First-base-side camera (x > 0) looking across the flight at its midpoint, far enough back that
 * release and plate fit the horizontal field of view at `aspect` with 15% spare each side.
 * @param path {t, p} exported path (Statcast feet); release is p[0], the plate the last sample.
 * @param backY optional Statcast y (ft) the frame must also reach behind release, e.g. the rubber
 *   so a pitcher's windup (M5 delivery) is in shot.
 * @returns {{position: number[], target: number[]}} Statcast feet; use vertical FOV SIDE_VFOV_DEG.
 */
export function sideView(path, aspect = 16 / 9, backY = null) {
  const a = path.p[0];
  const b = path.p[path.p.length - 1];
  const far = backY == null ? a[1] : Math.max(a[1], backY);
  const midY = (far + b[1]) / 2;
  const halfSpan = (Math.abs(far - b[1]) / 2) * 1.15;
  const hfov = 2 * Math.atan(Math.tan((SIDE_VFOV_DEG * Math.PI) / 360) * aspect);
  return {
    position: [halfSpan / Math.tan(hfov / 2), midY, SIDE_HEIGHT_FT],
    target: [0, midY, SIDE_HEIGHT_FT - 1],
  };
}

/**
 * Camera riding with the ball from release to the zone. At t = 0 it is over the pitcher's
 * throwing shoulder: behind the release point (larger y; the ball flies toward -y), above it and
 * out to the throwing side. It then follows the ball, 6 ft back along the flight direction
 * (central difference of the samples) and 1.5 ft up, the sideways offset easing to 0 by the plate,
 * always looking at the middle of the zone.
 * @param release [x, y, z] Statcast release point (ft); unused (the path starts at release), kept
 *   for the plan's interface. throws "R" or "L".
 * @param windup optional {pelvis: [x, y, z]} (Statcast ft, the pitcher's pelvis at t_s): with it,
 *   t_s < 0 gives the windup follow camera; without it, t_s < 0 is the same as t_s = 0.
 * @returns {{position: number[], target: number[]}} Statcast feet; use vertical FOV RIDE_VFOV_DEG.
 */
export function rideAlong(path, t_s, release, throws, windup = null) {
  if (windup && t_s < 0) {
    // Follow camera on the pitcher, eased into the t = 0 release viewpoint over the last WINDUP_BLEND_S.
    const side = throws === "L" ? 1 : -1;
    const [px, py, pz] = windup.pelvis;
    const w = {
      position: [px + side * WINDUP_OUT_FT, py + WINDUP_BEHIND_FT, pz + WINDUP_ABOVE_FT],
      target: [px, py - WINDUP_AIM_AHEAD_FT, WINDUP_AIM_Z_FT],
    };
    const r = rideAlong(path, 0, release, throws);
    const x = Math.min(1, Math.max(0, 1 + t_s / WINDUP_BLEND_S));
    const e = x * x * x * (x * (6 * x - 15) + 10); // smootherstep
    const mix = (a, b) => a.map((v, i) => v + (b[i] - v) * e);
    return { position: mix(w.position, r.position), target: mix(w.target, r.target) };
  }
  t_s = Math.max(t_s, 0); // the ball does not exist before release
  const h = 0.02;
  const p0 = stateAt(path, t_s - h).p;
  const p1 = stateAt(path, t_s + h).p;
  // The path itself throughout (not Statcast's release_pos, which differs slightly from the fit
  // at release and would make the camera jump as playback starts).
  const ball = stateAt(path, t_s).p;
  const d = p1.map((v, i) => v - p0[i]);
  const n = Math.hypot(...d) || 1;
  const dir = d.map((v) => v / n);
  // Throwing side: a righty's arm is on the catcher's left (x < 0) in Statcast axes.
  const out = (throws === "L" ? 1 : -1) * RIDE_SHOULDER_OUT_FT * (1 - Math.min(1, t_s / duration(path)));
  return {
    position: [ball[0] - RIDE_BEHIND_FT * dir[0] + out, ball[1] - RIDE_BEHIND_FT * dir[1],
      ball[2] - RIDE_BEHIND_FT * dir[2] + RIDE_ABOVE_FT],
    target: RIDE_AIM,
  };
}

// TV broadcast view: the centre-field camera, calibrated (2026-11-14) on two real centre-field frames
// of Ohtani pitching (Dodger Stadium and Houston; 800x450). Measured there, as fractions of the frame:
// pitcher's feet 0.76/0.85 down, his height 0.41/0.38 of the frame, the plate 0.55/0.50 down and
// 0.20/0.03 of the width to his right.
// - Position: on a platform just behind the outfield fence (real park cameras sit at the top of the
//   centre-field wall / batter's eye, ~400-420 ft out). The eye wall stands further back, behind it.
// - Side: the pitcher stands left of the plate on screen, so the camera is off to the third-base side
//   (x < 0, "left-centre"): nearer things shift away from the camera's offset. The shift between
//   pitcher (~350 ft away) and plate (~410 ft) is offset x (1/350 - 1/410) rad: the frames' 0.20 and
//   0.03 W give 43 and 7 ft depending on the park; 22 ft is a typical one (~0.11 W).
// - Height: likewise the plate sits above the pitcher's feet by height x (1/350 - 1/410) rad: the
//   frames give 24-42 ft; 32 ft (24 above the 8 ft fence) puts the plate ~0.24 H above his feet.
// - Aim: 20 ft in front of the plate, 2.5 ft up: the frame centre falls between the pitcher and the
//   plate, as on TV (feet ~0.8 H down, plate ~0.56 H).
// - Vertical FOV: a 6.3 ft pitcher at ~350 ft is 1.03 deg tall and fills ~0.4 of the frame -> 2.6 deg
//   (a ~650 mm lens on 16:9); the telephoto flattening stacks the plate just above his shoulders.
export const TV_VFOV_DEG = 2.6;
const TV_X_FT = -22;
const TV_Y_FT = fenceDistFt(0) + 10;
const TV_Z_FT = FENCE_HEIGHT_FT + 24;
const TV_AIM = [0, 20, 2.5];
// Near plane: 24-bit depth resolves ~d^2 / (near x 2^24); at the pitcher (~350 ft) the default 0.05 ft
// near gives ~0.15 ft, so the figure's overlapping parts z-fight. 100 ft gives ~7e-5 ft. Nothing
// visible is within 100 ft of this camera (it sits behind the fence, 400+ ft from the field).
export const TV_NEAR_FT = 100;
export const DEFAULT_NEAR_FT = 0.05; // close views (the sweep flies past the stands) need it small
/** Near plane (ft) for a pose: its own `near`, else the default. */
export const nearFor = (pose) => pose?.near ?? DEFAULT_NEAR_FT;

/**
 * Centre-field TV broadcast camera, the same for both pitcher hands.
 * @returns {{position: number[], target: number[], vfovDeg: number}} Statcast feet.
 */
export function tvView() {
  return { position: [TV_X_FT, TV_Y_FT, TV_Z_FT], target: TV_AIM, vfovDeg: TV_VFOV_DEG, near: TV_NEAR_FT };
}

// Free orbit (OrbitControls, in main.js): first-base side behind the plate, high, looking at mid-flight.
export const FREE_START = { position: [35, -18, 16], target: [0, 27, 4] };
export const FREE_MIN_DIST_FT = 2;
export const FREE_MAX_DIST_FT = 2200; // out to the whole floating island (~1300 ft across); far plane is 6000

/*
 * Intro sweep (the cricket clip's wide-to-low move): from high over the whole floating island down
 * to the umpire's spot, played on load and on a key. A path in polar coordinates about home plate:
 * - distance r falls geometrically (log-linear) from 1400 ft to the umpire's 5 ft, so the move
 *   feels even at every scale (each second covers the same *ratio* of distance);
 * - height falls likewise but on e^3, so it stays high until the last third and only drops once
 *   it is inside the stands (it must clear the 93 ft roof behind home, then the 32 ft net);
 * - the bearing swings from the left-field side round to straight behind home, settling early;
 * - the aim slides from the middle of the park to the umpire's aim point.
 * e is a smootherstep of t / INTRO_S (zero speed and acceleration at both ends).
 */
export const INTRO_S = 9;
const INTRO_START = { rFt: 1400, hFt: 650, bearingDeg: 300, vfovDeg: 50, aim: [0, 170, 0] };

/**
 * Camera pose t s into the intro sweep (clamped to [0, INTRO_S]); at INTRO_S it is umpireView().
 * @returns {{position: number[], target: number[], vfovDeg: number}} Statcast feet.
 */
export function introSweep(t) {
  const u = umpireView();
  const x = Math.min(1, Math.max(0, t / INTRO_S));
  const e = x * x * x * (x * (6 * x - 15) + 10); // smootherstep
  if (e >= 1) return u;
  const r0 = Math.hypot(u.position[0], u.position[1]); // 5 ft behind the plate
  const r = INTRO_START.rFt * (r0 / INTRO_START.rFt) ** e;
  const h = INTRO_START.hFt * (u.position[2] / INTRO_START.hFt) ** (e ** 3);
  const bearing = ((180 + (INTRO_START.bearingDeg - 180) * (1 - e) ** 2) * Math.PI) / 180;
  const k = e ** 1.5;
  return {
    position: [r * Math.sin(bearing), r * Math.cos(bearing), h],
    target: INTRO_START.aim.map((v, i) => v + (u.target[i] - v) * k),
    vfovDeg: INTRO_START.vfovDeg + (u.vfovDeg - INTRO_START.vfovDeg) * e,
  };
}
