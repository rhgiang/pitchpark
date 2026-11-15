// Viewer state as a pure reducer (no Three.js, no DOM) so it tests in Node.
// The data is one pitcher's export (scripts/export_viewer.py): held-out at-bats, pitches in order.
import { duration } from "./trajectory.js";

export const MODES = ["real", "sim", "ghost"]; // "Statcast fit", "M1 + M3", both
export const SPEEDS = [1, 0.25, 0.1];
export const VIEWS = ["umpire", "batter", "side", "ride", "free"]; // keys 1-5

// Ball rotation on screen. True spin is ~40 rev/s, which a 60 Hz screen cannot show at 1x (it
// would alias), so at 1x we show 1/20 of it; in slow motion the true rate is shown (angle =
// true rate x flight time, and slow motion stretches the flight time on screen).
export const SLOWED_SPIN_FACTOR = 1 / 20;
export const spinFactor = (speed) => (speed >= 1 ? SLOWED_SPIN_FACTOR : 1);

export function initialState() {
  return {
    slug: null, data: null, delivery: null, abIndex: 0, pitchIndex: 0,
    t: 0, playing: false, speed: 1, view: "umpire",
    showNoSpin: false, // the "no spin" reference path (export's nospin), toggled with N
    seen: 0, // pitches of this at-bat whose result may show: landed or passed (see revealed)
    wantMode: "real", // what the user picked; kept across pitches that cannot show it
    mode: "real", // what is shown: wantMode, or "real" when modeAllowed is false
    modeAllowed: false,
  };
}

export const currentAtBat = (s) => s.data?.at_bats[s.abIndex] ?? null;
export const currentPitch = (s) => currentAtBat(s)?.pitches[s.pitchIndex] ?? null;

/**
 * Seconds of delivery before release (t < 0), 0 for a pitcher with no delivery. The windup is part
 * of the timeline; release stays t = 0 so the flight, ?t= and the tests keep their meaning.
 */
export const preroll = (s) => s.delivery?.pre_s ?? 0;
// (callers use 0 - preroll, not -preroll: -0 would fail strict equality with 0)

/** Paths the current mode flies: [] for no_tracking; real only unless the pitch is modelled. */
export function shownPaths(s) {
  const p = currentPitch(s);
  if (!p?.real) return [];
  return { real: [p.real], sim: [p.sim], ghost: [p.real, p.sim] }[s.mode];
}

/** End of the current flight in s: the longest shown path (ghost: real or sim), 0 if none. */
export function flightEnd(s) {
  return Math.max(0, ...shownPaths(s).map(duration));
}

/**
 * Has the current pitch landed? Gates the result, at-bat event and "in the dirt" note so the HUD
 * never spoils a pitch before it arrives. A pitch with no flight (no_tracking) counts as landed,
 * and so does one already seen (a mode switch must not hide the result again).
 */
export function landed(s) {
  return shownPaths(s).length === 0 || (s.t > 0 && s.t >= flightEnd(s)) || s.seen > s.pitchIndex;
}

/**
 * May pitch i of the current at-bat show its result (pitch-strip colour)? Only once it has
 * landed or been passed; going back to an earlier pitch keeps what was already seen.
 */
export const revealed = (s, i) => i < s.seen;

export function reduce(s, a) {
  const next = step(s, a);
  if (next === s || !next.data) return next;
  const seen = next.pitchIndex + (landed(next) ? 1 : 0);
  return seen > next.seen ? { ...next, seen } : next;
}

function step(s, a) {
  switch (a.type) {
    case "pickPitcher":
      return derive({
        ...initialState(), speed: s.speed, view: s.view, showNoSpin: s.showNoSpin,
        slug: a.slug, data: a.data, delivery: a.delivery ?? null,
        t: 0 - (a.delivery?.pre_s ?? 0),
      });
    case "pickAtBat": {
      const n = s.data.at_bats.length;
      const abIndex = a.index === "random" ? Math.floor(mulberry32(a.seed) * n) : a.index;
      return derive({ ...s, abIndex, pitchIndex: 0, t: 0 - preroll(s), playing: false, wantMode: "real", seen: 0 });
    }
    case "goPitch": { // a pitch-strip click: that pitch, thrown from release
      const last = currentAtBat(s).pitches.length - 1;
      return derive({ ...s, pitchIndex: Math.min(last, Math.max(0, a.index)), t: 0 - preroll(s), playing: true });
    }
    case "nextPitch":
    case "prevPitch": {
      const last = currentAtBat(s).pitches.length - 1;
      const i = Math.min(last, Math.max(0, s.pitchIndex + (a.type === "nextPitch" ? 1 : -1)));
      return derive({ ...s, pitchIndex: i, t: 0 - preroll(s), playing: true });
    }
    case "replay":
      return { ...s, t: 0 - preroll(s), playing: true };
    case "seek": // frozen at t (clamped to the flight), e.g. ?t= for screenshots
      return { ...s, t: Math.min(flightEnd(s), Math.max(0 - preroll(s), a.t)), playing: false };
    case "toggleNoSpin":
      return { ...s, showNoSpin: !s.showNoSpin };
    case "setMode": // unknown values (e.g. a bad ?mode=) are ignored
      return MODES.includes(a.mode) ? derive({ ...s, wantMode: a.mode }) : s;
    case "setView": // unknown values (e.g. a bad ?view=) are ignored; pitch and t are kept
      return VIEWS.includes(a.view) ? { ...s, view: a.view } : s;
    case "setSpeed":
      return SPEEDS.includes(a.speed) ? { ...s, speed: a.speed } : s;
    case "tick": {
      if (!s.playing) return s;
      // Slow motion scales flight time only; the samples themselves are untouched.
      const end = flightEnd(s);
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
