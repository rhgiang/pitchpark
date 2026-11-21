// "Call the pitches" scoring and the reveal card's text. Pure (no DOM): ui.js draws it.
import { isStrike } from "./engine.js";

/**
 * Score a ball/strike call against the zone call at the plate (engine.isStrike, the same rule the
 * hitter model and the count use), not against the hitter's outcome: a swung-at ball is still a ball.
 * No call is "skipped" and is not counted. `ctx` = {face, stand} (the zone depends on the stand).
 */
export function scoreCall(call, pitch, { face, stand }) {
  if (call == null) return "skipped";
  const truth = isStrike(face, stand, pitch.plate_sim) ? "strike" : "ball";
  return call === truth ? "right" : "wrong";
}

/** Score a pitch-type guess (the code, e.g. "SL"). */
export function scoreType(pick, pitch) {
  if (pick == null) return "skipped";
  return pick === pitch.pitch_type ? "right" : "wrong";
}

/** scores = [{call, type}, ...] of "right" | "wrong" | "skipped" -> {calls: [right, total], types: [right, total]}; skipped is not counted. */
export function tally(scores) {
  const out = { calls: [0, 0], types: [0, 0] };
  for (const s of scores) for (const k of ["call", "type"]) {
    if (s[k] === "skipped") continue;
    out[`${k}s`][1]++;
    if (s[k] === "right") out[`${k}s`][0]++;
  }
  return out;
}

/** "1-2 vs RHB: 41% SL": how likely the model was to pick this type in this count. */
export const whyText = (p) => `${p.why.count} vs ${p.why.stand}HB: ${Math.round(p.why.p * 100)}% ${p.pitch_type}`;

const cap = (s) => s.charAt(0).toUpperCase() + s.slice(1);
const words = (s) => cap(s.replaceAll("_", " "));

/** "Called strike · Strikeout looking": Statcast words, then how the at-bat ended (if it did). */
export function resultLine(p) {
  if (!p.events) return words(p.description);
  const looking = p.description === "called_strike";
  const ev = p.events === "strikeout" ? `Strikeout ${looking ? "looking" : "swinging"}` : words(p.events);
  return `${words(p.description)} · ${ev}`;
}
