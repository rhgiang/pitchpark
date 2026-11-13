import { stateAt, duration } from "./trajectory.js";

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
const SIDE_HEIGHT_FT = 5; // side camera height; it aims 1 ft lower (framing choices)

/**
 * First-base-side camera (x > 0) looking across the flight at its midpoint, far enough back that
 * release and plate fit the horizontal field of view at `aspect` with 15% spare each side.
 * @param path {t, p} exported path (Statcast feet); release is p[0], the plate the last sample.
 * @returns {{position: number[], target: number[]}} Statcast feet; use vertical FOV SIDE_VFOV_DEG.
 */
export function sideView(path, aspect = 16 / 9) {
  const a = path.p[0];
  const b = path.p[path.p.length - 1];
  const midY = (a[1] + b[1]) / 2;
  const halfSpan = (Math.abs(a[1] - b[1]) / 2) * 1.15;
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
 * @returns {{position: number[], target: number[]}} Statcast feet; use vertical FOV RIDE_VFOV_DEG.
 */
export function rideAlong(path, t_s, release, throws) {
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
