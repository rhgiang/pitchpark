// The park's plan view, in Statcast feet (x toward the catcher's right = first-base side, y from
// home plate toward the pitcher). Real where it matters (OBR 2.01, Appendix 2): 90 ft bases,
// foul lines at 45 deg from the back point of home plate. The rest (fence distances, foul
// territory, stand sizes) is art, chosen to look like a typical MLB park.

export const BASE_FT = 90;
const D = BASE_FT / Math.SQRT2; // 63.64 ft: a base's x and y
export const BASES = { first: [D, D], second: [0, 2 * D], third: [-D, D] };
export const BASE_SIZE_FT = 18 / 12; // 18 in square bases (MLB since 2023)
export const MOUND_CENTRE_Y_FT = 59;

// Outfield fence: 330 ft down the lines, 375 in the gaps, 400 to centre (common MLB values).
// d(a) = 400 - 110 a^2 + 40 a^4, a = |angle from the centre-field line| / 45 deg, passes through
// all three exactly and falls monotonically to the poles.
export const FENCE_HEIGHT_FT = 8;
export function fenceDistFt(angleDeg) {
  const a = Math.abs(angleDeg) / 45;
  return 400 - 110 * a * a + 40 * a ** 4;
}

/** Statcast [x, y] at `angleDeg` from the centre-field line (+ toward first base), r ft out. */
export const polar = (angleDeg, r) => {
  const t = (angleDeg * Math.PI) / 180;
  return [r * Math.sin(t), r * Math.cos(t)];
};

const add = (a, b, k = 1) => [a[0] + k * b[0], a[1] + k * b[1]];
const U_R = [Math.SQRT1_2, Math.SQRT1_2]; // along the first-base line
const N_R = [Math.SQRT1_2, -Math.SQRT1_2]; // perpendicular to it, into foul territory
export const POLE_R = polar(45, fenceDistFt(45));
export const POLE_L = polar(-45, fenceDistFt(45));

/** Resample a polyline at a fixed spacing (ft). Keeps the first and last points. */
export function resample(pts, spacing) {
  const out = [pts[0]];
  let carry = 0;
  for (let i = 1; i < pts.length; i++) {
    const [ax, ay] = pts[i - 1], [bx, by] = pts[i];
    const len = Math.hypot(bx - ax, by - ay);
    let s = spacing - carry;
    while (s <= len) {
      out.push([ax + ((bx - ax) * s) / len, ay + ((by - ay) * s) / len]);
      s += spacing;
    }
    carry = len - (s - spacing);
  }
  const last = pts[pts.length - 1];
  const prev = out[out.length - 1];
  if (Math.hypot(last[0] - prev[0], last[1] - prev[1]) > spacing * 0.3) out.push(last);
  else out[out.length - 1] = last;
  return out;
}

/** Unit normals of a polyline, on its left (the outside, for paths that run clockwise). */
export function normals(pts) {
  return pts.map((_, i) => {
    const a = pts[Math.max(i - 1, 0)], b = pts[Math.min(i + 1, pts.length - 1)];
    const tx = b[0] - a[0], ty = b[1] - a[1];
    const n = Math.hypot(tx, ty) || 1;
    return [-ty / n, tx / n];
  });
}

/** Points along the outfield fence, left-field pole to right-field pole (clockwise). */
export function fencePath(step = 1) {
  const pts = [];
  for (let a = -45; a <= 45 + 1e-9; a += step) pts.push(polar(a, fenceDistFt(a)));
  return pts;
}

// Foul-territory wall, first-base side: from the pole, a short corner 15 ft into foul ground,
// then a wall tapering out to 60 ft off the line at 120 ft from home, parallel to the line from
// there in, and a 60 ft arc behind home plate (OBR recommends >= 60 ft to the backstop).
const FOUL_CORNER = add(POLE_R, N_R, 15);
const FOUL_KNEE = add(add([0, 0], U_R, 120), N_R, 60);
const FOUL_HOME = add([0, 0], N_R, 60); // on the 60 ft backstop circle, at 135 deg
export const BACKSTOP_FT = 60;
const mirror = ([x, y]) => [-x, y];

/**
 * The front of the main stands: right-field corner, along the first-base side, round behind
 * home, out to the left-field corner (clockwise seen from above, so normals() points outward).
 */
export function foulWallPath() {
  const right = [POLE_R, FOUL_CORNER, FOUL_KNEE, FOUL_HOME];
  const arc = [];
  for (let a = 135; a <= 225 + 1e-9; a += 3) arc.push(polar(a, BACKSTOP_FT));
  const left = [...right].reverse().map(mirror);
  return [...right, ...arc.slice(1, -1), ...left];
}

/** The playing surface's closed boundary (fence then foul walls), clockwise. */
export function fieldBoundary() {
  return [...fencePath(1), ...foulWallPath().slice(1, -1)];
}

/** Even-odd point-in-polygon test, [x, y] against a closed polyline. */
export function inside(p, poly) {
  let c = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [xi, yi] = poly[i], [xj, yj] = poly[j];
    if ((yi > p[1]) !== (yj > p[1]) && p[0] < ((xj - xi) * (p[1] - yi)) / (yj - yi) + xi) c = !c;
  }
  return c;
}

/** Seeded PRNG (mulberry32): the park's scatter is the same on every load. */
export function rng(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Middle of the bowl: light towers aim here; look.js's night falloff and sun are centred here.
export const FIELD_CENTRE_FT = [0, 170];
// The floating island the park sits on (art): a disc around the middle of the park.
export const ISLAND = { centre: [0, 165], radiusFt: 560 };
// Outfield: bleachers start this far behind the fence; the gap holds the bullpens.
export const BLEACHER_GAP_FT = 32;
export const BATTERS_EYE_HALF_DEG = 9; // centre-field batter's eye: no seats in |angle| < 9 deg
