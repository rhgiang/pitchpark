// Camera placements in Statcast feet (x catcher's right, y plate -> pitcher, z up).

// Batter's box: 4 ft wide x 6 ft long, inner line 6 in from the 17 in plate (Official Baseball
// Rules 2.01 diagram), so the box spans |x| = 14.5 in to 62.5 in and is centred on the plate's
// middle (y ~ 0.71 ft), running from about y = -2.3 to 3.7 ft.
// Eye height: standing eye height is ~0.936 x stature (Drillis & Contini 1966, segment
// proportions); for an MLB-average ~6 ft 2 in hitter that is ~5.8 ft, less ~9 in of batting
// crouch -> 5.0 ft. The crouch amount is our estimate, not a measurement.
export const EYE_HEIGHT_FT = 5.0;
// Head over the inner half of the box, leaning toward the plate: |x| = 2.5 ft (our choice,
// inside the box's 1.2-5.2 ft span).
export const STANCE_X_FT = 2.5;
// Toward the back of the box (y = -2.3 at the back line), where many hitters set up for time.
// Also what lets the zone and the pitcher fit in one frame. Our choice.
export const STANCE_Y_FT = -2.0;

// Aim point at the front of the plate, 2.0 ft up: below the middle of a typical 1.6-3.4 ft zone
// so the whole zone, bottom included, stays in frame.
const ZONE_AIM = [0, 17 / 12, 2.0];
const RELEASE_AREA = [0, 55, 6]; // roughly where pitchers let go

/**
 * Batter's-eye camera for batter side `stand` ("R" or "L").
 * A righty stands on the catcher's left (x < 0), a lefty on the right.
 * Aims along the bisector of the directions to the strike zone and to the release area, so both
 * are in frame (needs a ~50 deg vertical FOV: the zone sits up to ~35 deg below the eye line).
 * @returns {{position: number[], target: number[]}} Statcast feet.
 */
export function batterEye(stand) {
  const side = stand === "L" ? 1 : -1;
  const position = [side * STANCE_X_FT, STANCE_Y_FT, EYE_HEIGHT_FT];
  const a = unit(sub(ZONE_AIM, position));
  const b = unit(sub(RELEASE_AREA, position));
  const dir = unit(a.map((v, i) => v + b[i]));
  return { position, target: position.map((v, i) => v + 10 * dir[i]) };
}

const sub = (u, v) => u.map((a, i) => a - v[i]);
const unit = (u) => {
  const n = Math.hypot(...u);
  return u.map((a) => a / n);
};
