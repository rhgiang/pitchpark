// Pure logic for the "something went wrong / running slow / small screen" cards (DOM: notices.js).
import { displayName } from "./menu.js";

export const LITE_KEY = "pitchpark.lite.v1";
export const PHONE_KEY = "pitchpark.phone.v1";

/** Storage that may be missing or throw (private window, blocked site data). */
const defaultStorage = () => { try { return globalThis.localStorage ?? null; } catch { return null; } };

/** A saved on/off flag; any storage failure reads as off. */
export function getFlag(key, storage = defaultStorage()) {
  try { return storage?.getItem(key) === "1"; } catch { return false; }
}
/** Save a flag; a failure is ignored (the choice just is not remembered). */
export function setFlag(key, on, storage = defaultStorage()) {
  try { on ? storage?.setItem(key, "1") : storage?.removeItem(key); } catch { /* not remembered */ }
}

/** Can this browser make a WebGL context? `createCanvas` is injectable for tests. */
export function canMakeWebGL(createCanvas = () => document.createElement("canvas")) {
  try {
    const gl = createCanvas().getContext("webgl2") ?? createCanvas().getContext("webgl");
    gl?.getExtension?.("WEBGL_lose_context")?.loseContext?.(); // hand the probe's context back
    return !!gl;
  } catch { return false; }
}

export const SLOW_FPS = 28; // below this counts as slow (a healthy 30 Hz display still measures ~30)
export const SLOW_WINDOW_MS = 5000; // of ACTIVE playback time, not wall time
const SLOW_GAP_MS = 1000; // a longer gap between frames is a stall (tab hidden, shader compile), not a rate

/**
 * Frame-rate watcher. Call `frame(now, active)` once per display frame; it reports true once
 * SLOW_WINDOW_MS of active frames averaged below SLOW_FPS. Idle frames (paused, menu, hidden tab)
 * are not counted: pacing.js deliberately idles at 20 fps, which is not "slow". Active time adds up
 * across short stretches (one pitch is only ~3 s), and a stall gap is skipped rather than averaged in.
 */
export function makeSlowDetector({ minFps = SLOW_FPS, windowMs = SLOW_WINDOW_MS } = {}) {
  let prev = null, frames = 0, ms = 0;
  return {
    frame(now, active) {
      if (!active) { prev = null; return false; }
      if (prev !== null && now - prev <= SLOW_GAP_MS) { frames++; ms += now - prev; }
      prev = now;
      if (ms < windowMs) return false;
      const fps = (frames * 1000) / ms;
      frames = ms = 0;
      return fps < minFps;
    },
  };
}

/** A phone: touch-first and a small screen (smaller side, so landscape counts). */
export const isPhone = ({ coarse, width, height }) => !!coarse && Math.min(width, height) <= 500;

/** Is this slug one of the park's pitchers? */
export const slugInIndex = (index, slug) => index.some((p) => p.slug === slug);

/** Pitchers (other than `current`) that have a comparison, as {slug, name}, in index order. */
export function compareChoices(index, available, current) {
  return index.filter((p) => available.has(p.slug) && p.slug !== current).map((p) => ({ slug: p.slug, name: displayName(p.name) }));
}

/** Names for debug switches: ?debug=slow,nowebgl (dev server only; see debug.js). */
export const debugList = (search) => (new URLSearchParams(search).get("debug") ?? "").split(",").filter(Boolean);
