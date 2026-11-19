import { isOld } from "./old.js";
import { polar, fenceDistFt, resample, POLE_R } from "./layout.js";

/*
 * Pitch Park's signature features (toon, low-poly, instanced; each hidden by ?old=<name> for A/B):
 *   ?old=tower   the Spin Tower: a giant ball on a pylon behind centre field, twin light towers
 *   ?old=falls   the Falls: a cliff and waterfall into a pool beyond the right-field wall
 *   ?old=oracle  a tall brick right-field wall with arches, pennant poles, palms past left field
 * The layout here is pure (plain numbers, no three.js objects), so tests can check it and the
 * park plan can keep its trees, paths and tents out of the way (clearForSignature).
 * Statcast feet: x toward first base, y out from home plate.
 */

export const TOWER = {
  x: 0, y: 690, // on the centre-field line, near the island's rim (it is 560 ft from the island centre at y = 165)
  ballR: 13.5, ballZ: 71, // the ball: 27 ft across, 71 ft up
  twinX: 80, mastH: 68, // the two light towers either side
};
// The umpire's frame reaches +7.3 deg above the horizon; the batter's eye wall (36 ft at 426 ft) hides everything below
// 4.2 deg. The ball (radius 14.5 ft at 695 ft from the plate) is sized to sit exactly in that 3-degree band.
export const EYE_PANEL = { w: 100, h: 24, z: 22 }; // the rpm readout, on the face of the batter's eye wall

/** Falls: bearings (deg from the centre-field line, + toward first base) and ft behind the fence. */
export const FALLS = {
  a0: 22, a1: 45, // the brick wall (from the right-field bleachers to the foul pole) and the pool span these bearings
  aCliff0: 12, aCliff1: 48, aFall: 30, // the cliff's footprint; the water falls at aFall
  wallOff: 1.8, // the brick wall stands this far behind the padded fence: it IS the right-field wall
  poolIn: 44, poolOut: 122, // the pool: ft behind the fence
  cliffOff: 122, // the cliff's foot
  cliffH: 96, // height of the cliff's top
  faceRun: 46, // horizontal run of the steep face
};

/** The brick wall: bay width, lower (solid, over the 8 ft padded fence) and upper (arcade) storey heights, thickness, arch opening width and spring height, sill height, walkway depth (ft). */
export const WALL = { bay: 22, h1: 18, h2: 22, thick: 3, openW: 14, spring: 12, sill: 3.5, deck: 9 };
export const WALL_TOP_FT = WALL.h1 + WALL.h2; // 40; the cornice adds 3.2

/** The pool's footprint and the cliff's, as plan test: is p = [x, y] within bearings a0..a1 and beyond `off` ft behind the fence? */
export function inSector(p, a0, a1, off = 0) {
  const r = Math.hypot(p[0], p[1]);
  const a = (Math.atan2(p[0], p[1]) * 180) / Math.PI;
  return a >= a0 && a <= a1 && r > fenceDistFt(a) + off;
}

/**
 * Wall bays, one {p, yaw, len, ret?} each, from the right-field bleachers (22 deg) along the fence to the foul pole, then on
 * round the corner 14 ft into foul ground where the stands begin (the wall is continuous, with no free ends), plus
 * a short return (ret: true) across the bullpen gap at the bleachers' end. The array's middle bay is the centre piece.
 */
export function wallBays() {
  const pts = [];
  for (let a = FALLS.a0; a <= FALLS.a1 + 1e-9; a += 0.25) pts.push(polar(a, fenceDistFt(a) + FALLS.wallOff));
  const k = FALLS.wallOff, n45 = [Math.SQRT1_2, -Math.SQRT1_2]; // radial offset at the pole, then 14 ft into foul ground
  pts.push([POLE_R[0] + Math.SQRT1_2 * k + n45[0] * 14, POLE_R[1] + Math.SQRT1_2 * k + n45[1] * 14]);
  const r = resample(pts, WALL.bay);
  const bays = r.slice(1).map((q, i) => {
    const p = r[i], dx = q[0] - p[0], dy = q[1] - p[1];
    return { p: [(p[0] + q[0]) / 2, (p[1] + q[1]) / 2], yaw: Math.atan2(dy, dx), len: Math.hypot(dx, dy) };
  });
  // The return: radially out from the wall's first point to the bleachers' front (32 ft behind the fence).
  const t = (FALLS.a0 * Math.PI) / 180, a0 = pts[0], len = 32 - FALLS.wallOff;
  bays.push({ p: [a0[0] + (Math.sin(t) * len) / 2, a0[1] + (Math.cos(t) * len) / 2], yaw: Math.atan2(Math.cos(t), Math.sin(t)), len, ret: true });
  return bays;
}

/** Palm sites beyond left field: [{p, h, yaw, lean}], a fixed pattern (no randomness: it is the same every load). */
export function palmSites() {
  const out = [];
  let k = 0;
  for (let a = -46; a <= -12; a += 2.4, k++) {
    // two staggered rows behind the left bleachers' back wall (fence + 71 ft)
    const off = 96 + (k % 3) * 17 + Math.sin(k * 2.1) * 6;
    out.push({ p: polar(a + (k % 2) * 0.8, fenceDistFt(a) + off), h: 36 + ((k * 7) % 5) * 3.5, yaw: k * 1.9, lean: 0.08 + ((k * 3) % 4) * 0.025 });
  }
  return out;
}

/** The spin readout: "2,277" for 2277.4 rpm, and "---" when the pitch has no spin data. */
export function rpmLabel(rpm) {
  return Number.isFinite(rpm) ? Math.round(rpm).toLocaleString("en-US") : "---";
}

/** Ball rotation angle (rad) about its spin axis t_s seconds into the flight; factor slows it (state.js spinFactor). */
export const spinAngleRad = (rpm, t_s, factor = 1) => 2 * Math.PI * (rpm / 60) * t_s * factor;

/**
 * Keep the plan's scatter (trees, bushes, rocks, lamps, benches, tents, the 40 deg path) out of the signature
 * features' footprints. Filters the plan in place; uses no random numbers, so the rest of the layout is unchanged.
 */
export function clearForSignature(plan) {
  const sectors = [];
  const rects = [];
  if (!isOld("falls")) sectors.push((p) => inSector(p, FALLS.aCliff0 - 2, FALLS.aCliff1 + 2, 28));
  if (!isOld("tower")) {
    rects.push([-150, 150, 470, 760]); // trees behind the ball would clutter the sky it is seen against
    rects.push([-24, 24, 600, 760]);
  }
  const near = (p) => sectors.some((f) => f(p)) || rects.some(([x0, x1, y0, y1]) => p[0] > x0 && p[0] < x1 && p[1] > y0 && p[1] < y1);
  const palms = isOld("oracle") ? [] : palmSites().map((s) => s.p);
  const nearPalm = (p) => palms.some((q) => Math.hypot(p[0] - q[0], p[1] - q[1]) < 15);
  const drop = (p) => near(p) || nearPalm(p);
  for (const key of ["trees", "groundTrees", "bushes", "rocks", "lamps", "benches", "tents"]) {
    if (plan[key]) plan[key] = plan[key].filter((s) => !drop(s.p));
  }
  if (!isOld("falls")) plan.paths = plan.paths.filter((path) => !path.some((q) => inSector(q, FALLS.aCliff0, FALLS.aCliff1, 28)));
}
