// Playback of exported paths. No physics here (Python is the source of truth): we only
// evaluate what the exporter wrote (schema 3: the Statcast fit's 9 parameters, and RK4 knots joined
// by cubic Hermite), so what flies is what Python computed. expandPitcher turns that into the
// {t, p} sample arrays (5 ms) that the ball, trail and cameras read.

export const SCHEMA = 3;

/** Flight time of a path {t: [...s], p: [[x,y,z] ft, ...]}, release (t = 0) to the plate, in s. */
export function duration(path) {
  return path.t[path.t.length - 1];
}

/**
 * Position at time t_s by linear interpolation between samples; clamped to the first/last
 * sample outside the path. With 5 ms samples the chord error is about a*dt^2/8, ~0.002 in
 * for a ~40 ft/s^2 pitch, far under the 0.01 in truth-test tolerance.
 * @returns {{p: number[], done: boolean}} done once t_s reaches the plate.
 */
export function stateAt(path, t_s) {
  const { t, p } = path;
  const n = t.length;
  if (t_s >= t[n - 1]) return { p: [...p[n - 1]], done: true };
  if (t_s <= t[0]) return { p: [...p[0]], done: false };
  let lo = 0;
  let hi = n - 1; // binary search: t[lo] <= t_s < t[hi]
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (t[mid] <= t_s) lo = mid;
    else hi = mid;
  }
  const f = (t_s - t[lo]) / (t[hi] - t[lo]);
  return { p: p[lo].map((a, i) => a + f * (p[hi][i] - a)), done: false };
}

/**
 * Where the visible flight ends: the first time the path's height drops to minZ_ft (linear
 * between samples; ball.js hides the ball below one radius), else the last sample (the plate).
 * Balls in the dirt: Statcast's fit carries on below the ground, but nobody sees that part.
 * @returns {{t: number, p: number[]}} time in s and position in Statcast feet.
 */
export function visibleEnd(path, minZ_ft) {
  const { t, p } = path;
  for (let k = 0; k < t.length; k++) {
    if (p[k][2] >= minZ_ft) continue;
    if (k === 0) return { t: t[0], p: [...p[0]] };
    const f = (p[k - 1][2] - minZ_ft) / (p[k - 1][2] - p[k][2]);
    return { t: t[k - 1] + f * (t[k] - t[k - 1]), p: p[k - 1].map((a, i) => a + f * (p[k][i] - a)) };
  }
  return { t: t[t.length - 1], p: [...p[p.length - 1]] };
}

/** Playback sample times: 0, dt, ... strictly before T, then T (the exporter's old 5 ms grid). */
function grid(T, dt) {
  const n = Math.ceil((T - 1e-9) / dt);
  const t = new Array(n + 1);
  for (let k = 0; k < n; k++) t[k] = Math.round(k * dt * 1e4) / 1e4;
  t[n] = T;
  return t;
}

/** The Statcast fit {f: [pos, vel, acc at y = 50 ft], r: release time on the fit's clock, T} as {t, p}. */
export function expandFit({ f, r, T }, dt) {
  const t = grid(T, dt);
  const p = t.map((s) => {
    const tau = r + s;
    return [0, 1, 2].map((k) => f[k] + f[3 + k] * tau + 0.5 * f[6 + k] * tau * tau);
  });
  return { t, p };
}

/**
 * RK4 knots {s: [[x, y, z, vx, vy, vz], ...], T} as {t, p}: cubic Hermite between knots (knot k
 * is at k * knot, the last at T). Error is about h^4 / 384 times the 4th derivative of position,
 * far under 0.01 ft at 50 ms (results/web-performance.md).
 */
export function expandKnots({ s, T }, knot, dt) {
  const t = grid(T, dt);
  const at = (j) => (j === s.length - 1 ? T : j * knot);
  let j = 0;
  const p = t.map((u) => {
    while (j < s.length - 2 && at(j + 1) <= u) j++;
    const h = at(j + 1) - at(j);
    const x = (u - at(j)) / h;
    const x2 = x * x, x3 = x2 * x;
    const h00 = 2 * x3 - 3 * x2 + 1, h10 = x3 - 2 * x2 + x, h01 = -2 * x3 + 3 * x2, h11 = x3 - x2;
    return [0, 1, 2].map((k) => h00 * s[j][k] + h10 * h * s[j][3 + k] + h01 * s[j + 1][k] + h11 * h * s[j + 1][3 + k]);
  });
  return { t, p };
}

/**
 * Schema-3 export -> the layout the viewer reads: each pitch's real / sim / nospin becomes a
 * {t, p} path, expanded on first access (most pitches are never played). Throws on any other
 * schema so the caller can show the data-error card instead of misreading old or newer files.
 */
export function expandPitcher(data) {
  const m = data?.meta;
  if (m?.schema !== SCHEMA) throw new Error(`pitch data schema ${m?.schema} is not supported (expected ${SCHEMA})`);
  const expanders = {
    real: (c) => expandFit(c, m.step_s),
    sim: (c) => expandKnots(c, m.knot_s, m.step_s),
    nospin: (c) => expandKnots(c, m.knot_s, m.step_s),
  };
  for (const ab of data.at_bats) {
    for (const pitch of ab.pitches) {
      for (const [key, expand] of Object.entries(expanders)) {
        const c = pitch[key];
        if (!c) continue;
        let path;
        Object.defineProperty(pitch, key, { enumerable: true, configurable: true, get: () => (path ??= expand(c)) });
      }
    }
  }
  return data;
}
