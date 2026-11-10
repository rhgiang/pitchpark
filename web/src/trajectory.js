// Playback of exported paths. No physics here (Python is the source of truth): we only
// interpolate the exporter's time-stamped samples, so what flies is what Python computed.

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
