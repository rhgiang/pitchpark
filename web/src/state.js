// Viewer state as a pure reducer (no Three.js, no DOM) so it tests in Node.
// The data is one pitcher's export (scripts/export_viewer.py): held-out at-bats, pitches in order.
import { duration } from "./trajectory.js";

export const MODES = ["real", "sim", "ghost"]; // "Statcast fit", "M1 + M3", both
export const SPEEDS = [1, 0.25, 0.1];

export function initialState() {
  return {
    slug: null, data: null, abIndex: 0, pitchIndex: 0,
    t: 0, playing: false, speed: 1,
    wantMode: "real", // what the user picked; kept across pitches that cannot show it
    mode: "real", // what is shown: wantMode, or "real" when modeAllowed is false
    modeAllowed: false,
  };
}

export const currentAtBat = (s) => s.data?.at_bats[s.abIndex] ?? null;
export const currentPitch = (s) => currentAtBat(s)?.pitches[s.pitchIndex] ?? null;

/** Paths the current mode flies: [] for no_tracking; real only unless the pitch is modelled. */
export function shownPaths(s) {
  const p = currentPitch(s);
  if (!p?.real) return [];
  return { real: [p.real], sim: [p.sim], ghost: [p.real, p.sim] }[s.mode];
}

export function reduce(s, a) {
  switch (a.type) {
    case "pickPitcher":
      return derive({ ...initialState(), speed: s.speed, slug: a.slug, data: a.data });
    case "pickAtBat": {
      const n = s.data.at_bats.length;
      const abIndex = a.index === "random" ? Math.floor(mulberry32(a.seed) * n) : a.index;
      return derive({ ...s, abIndex, pitchIndex: 0, t: 0, playing: false, wantMode: "real" });
    }
    case "nextPitch":
    case "prevPitch": {
      const last = currentAtBat(s).pitches.length - 1;
      const i = Math.min(last, Math.max(0, s.pitchIndex + (a.type === "nextPitch" ? 1 : -1)));
      return derive({ ...s, pitchIndex: i, t: 0, playing: true });
    }
    case "replay":
      return { ...s, t: 0, playing: true };
    case "setMode":
      return derive({ ...s, wantMode: a.mode });
    case "setSpeed":
      return { ...s, speed: a.speed };
    case "tick": {
      if (!s.playing) return s;
      const paths = shownPaths(s);
      // Slow motion scales flight time only; the samples themselves are untouched.
      const end = Math.max(0, ...paths.map(duration));
      const t = s.t + a.dt * s.speed;
      return t >= end ? { ...s, t: end, playing: false } : { ...s, t };
    }
    default:
      throw new Error(`unknown action ${a.type}`);
  }
}

function derive(s) {
  const modeAllowed = currentPitch(s)?.status === "ok";
  return { ...s, modeAllowed, mode: modeAllowed ? s.wantMode : "real" };
}

/** One uniform draw in [0, 1) from an integer seed (mulberry32, a small public-domain PRNG). */
function mulberry32(seed) {
  let x = (seed + 0x6d2b79f5) | 0;
  x = Math.imul(x ^ (x >>> 15), x | 1);
  x ^= x + Math.imul(x ^ (x >>> 7), x | 61);
  return ((x ^ (x >>> 14)) >>> 0) / 4294967296;
}
