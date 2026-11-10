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
// pitch, 2026-11-10) as the narrowest vertical FOV that, at 16:9 and within 95% of the frame,
// shows every pitch's zone corners, plate arrival (Statcast fit and simulated) and release point
// for the pitch's own batter side. Needs 66.6 deg; balls in the dirt (plate z down to -0.8 ft,
// 5.8 ft below the eye at ~3.7 ft away) set it. Checked by test/cameras.test.js.
export const AIM_YAW_DEG = 16.5; // turned toward the plate (right for a righty, left for a lefty)
export const AIM_PITCH_DEG = -26; // looking down
export const BATTER_EYE_VFOV_DEG = 67;

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

// Catcher's view (the main view): centred behind the plate at a crouched catcher's eye.
// Depth: the catcher must stay in the catcher's box until the pitch is released (OBR 5.02(a)); the
// box starts at the batter's boxes' back line, y ~ -2.3 ft (see above), so his feet are just
// behind it and his eyes, leaning forward over them, are at about y = -2.5 ft.
// Height: an estimate from Drillis & Contini (1966) segment proportions for a 6 ft 1 in catcher
// (MLB catchers average about that): in a deep crouch his hips are ~1.2 ft up (shins ~45 deg), and
// hip-to-eye is ~0.41 x stature = 2.5 ft, ~2.35 ft tall with the torso leaning ~20 deg forward,
// so the eyes are ~3.5 ft up. Not a measurement.
export const CATCHER_Y_FT = -2.5;
export const CATCHER_EYE_HEIGHT_FT = 3.5;
// Aim and FOV: chosen like the batter's eye (grid search over every exported pitch, 2026-11-10,
// at 16:9 within 95% of the frame: zone corners, release, and Statcast-fit, simulated and
// no-spin plate arrivals). The plate is only ~4 ft away, so the lowest arrivals (balls in the
// dirt, no-spin paths of sinkers and changeups) set the bottom edge and the release the top.
export const CATCHER_AIM_PITCH_DEG = -10.75;
export const CATCHER_VFOV_DEG = 77;

/**
 * Catcher's-eye camera, the same for both batter sides: on the plate's centre line (x = 0),
 * looking straight out at the pitcher, tilted down so the zone sits mid-frame.
 * @returns {{position: number[], target: number[], vfovDeg: number}} Statcast feet.
 */
export function catcherView() {
  const pitch = (CATCHER_AIM_PITCH_DEG * Math.PI) / 180;
  const position = [0, CATCHER_Y_FT, CATCHER_EYE_HEIGHT_FT];
  return {
    position,
    target: [0, CATCHER_Y_FT + 10 * Math.cos(pitch), CATCHER_EYE_HEIGHT_FT + 10 * Math.sin(pitch)],
    vfovDeg: CATCHER_VFOV_DEG,
  };
}

// Other views. Vertical FOVs are framing choices (the batter's eye needs 67 deg; these do not).
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
 * @param release [x, y, z] Statcast release point (ft); throws "R" or "L".
 * @returns {{position: number[], target: number[]}} Statcast feet; use vertical FOV RIDE_VFOV_DEG.
 */
export function rideAlong(path, t_s, release, throws) {
  const h = 0.02;
  const p0 = stateAt(path, t_s - h).p;
  const p1 = stateAt(path, t_s + h).p;
  // The ball, except at t = 0 where it is the release point itself (the path starts there).
  const ball = t_s <= 0 && release ? release : stateAt(path, t_s).p;
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
