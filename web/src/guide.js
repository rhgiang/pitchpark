// First-visit tour, "How it works" content and loading copy (pure; the DOM lives in guide-ui.js).
import { showMenu } from "./menu.js";

export const TOUR_KEY = "pitchpark.tour.v1";

/** Four tips; `target` is a CSS selector (first match that is visible wins), `html` is trusted static text. */
export const TOUR_STEPS = [
  { id: "seat", target: "#tabs", title: "Pick your seat",
    html: "Watch from the umpire, the batter's box, the side, riding with the ball, or the TV angle. Keys <kbd>1</kbd>–<kbd>6</kbd> switch views." },
  { id: "mode", target: "#mode-real", group: true, title: "Real, Sim, Ghost",
    html: "Real = the tracked pitch (Statcast). Sim = our physics flying it (M1+M3). Ghost = both at once (Both), to see the difference." },
  { id: "speed", target: "#speed-1", group: true, title: "Slow it down",
    html: "Speeds 1×, ¼×, ⅒× to watch the release and the break. Keys <kbd>Z</kbd> <kbd>X</kbd> <kbd>C</kbd>." },
  { id: "compare", target: "#help", title: "Compare to video",
    html: "The compare page lines the 3D delivery up with the real broadcast pose. Open this <b>?</b> panel for the link; it needs a clip from your own disk." },
].filter((s) => !(import.meta.env?.PROD && s.id === "compare")); // the public build has no compare page

/** Storage that may be missing or throw (private window, blocked site data). */
const defaultStorage = () => { try { return globalThis.localStorage ?? null; } catch { return null; } };

export function tourSeen(storage = defaultStorage()) {
  try { return storage?.getItem(TOUR_KEY) === "1"; } catch { return false; }
}
export function markTourSeen(storage = defaultStorage()) {
  try { storage?.setItem(TOUR_KEY, "1"); } catch { /* not remembered: it may show again, which is harmless */ }
}
export function clearTourSeen(storage = defaultStorage()) {
  try { storage?.removeItem(TOUR_KEY); } catch { /* nothing to clear */ }
}

/** Auto-show only for a first visit that came through the menu (a URL naming a pitcher is a deep link). */
export const shouldAutoTour = (search, seen) => showMenu(search) && !seen;

/** Next / previous step index, clamped; `null` after the last Next means done. */
export const nextStep = (i, n) => (i + 1 >= n ? null : i + 1);
export const prevStep = (i) => Math.max(0, i - 1);

/**
 * First step target that is on screen. `rectOf(selector)` returns a {left,top,width,height} or null;
 * a zero-size rect counts as hidden (display:none at phone width). Returns the padded rect or null.
 */
export function resolveTarget(selector, rectOf, pad = 6) {
  const r = rectOf(selector);
  if (!r || !(r.width > 0) || !(r.height > 0)) return null;
  return { left: r.left - pad, top: r.top - pad, width: r.width + 2 * pad, height: r.height + 2 * pad };
}

/**
 * Where the tip card goes: below the ring if it fits, else above, clamped to the viewport with a
 * margin; no ring -> centred.
 */
export function placeCard(ring, card, vw, vh, gap = 14, margin = 12) {
  const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
  const w = Math.min(card.width, vw - 2 * margin);
  if (!ring) return { left: (vw - w) / 2, top: Math.max(margin, (vh - card.height) / 2) };
  const left = clamp(ring.left + ring.width / 2 - w / 2, margin, vw - w - margin);
  const below = ring.top + ring.height + gap;
  const top = below + card.height + margin <= vh ? below : Math.max(margin, ring.top - gap - card.height);
  return { left, top };
}

/** "How it works": figures copied from results/ (held-out, median) -- see ux-batch-report.md for sources. */
export const HOW = {
  title: "How Pitch Park works",
  sub: "Real MLB pitches, re-flown by physics and thrown by a pitcher rebuilt from video.",
  steps: [
    { t: "Pitch tracking data", d: "Every pitch from Statcast: release point, speed, spin, movement.", n: "public data", c: "Baseball Savant" },
    { t: "Physics flight", d: "Gravity, air drag and spin (Magnus) fitted per pitch type, tested on games it never saw.", n: "2.0–4.0 in", c: "median miss, four-seamers (held-out games)" },
    { t: "Seam effects", d: "A learned push for the movement spin alone can't explain.", n: "22 of 22", c: "pitch types closer (median, held-out)" },
    { t: "Pitch-to-pitch variation", d: "A model that generates new pitches; a classifier tries to tell them from real ones.", n: "6 of 9", c: "graded groups indistinguishable" },
    { t: "The delivery", d: "Body pose from broadcast video, lifted to 3D and pinned to Statcast's release point.", n: "~2.1 in", c: "median 2D gap vs real clip, Ohtani" },
  ],
  disclaimer: "A personal research project; not affiliated with MLB. Pitch data from Baseball Savant (Statcast).",
};

export const loadingText = (name) => `Loading ${name}'s delivery…`;
export const errorText = (name) => `Couldn't load ${name}'s delivery.`;
