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
