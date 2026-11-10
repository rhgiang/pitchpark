import { currentAtBat, shownPaths } from "./state.js";
import { inDirt } from "./ball.js";

const MODE_LABEL = { real: "Statcast fit", sim: "M1 + M3", ghost: "Ghost: Statcast fit + M1 + M3 (translucent)" };

const fmt = (v, digits, unit = "") => (v == null ? "–" : `${v.toFixed(digits)}${unit}`);
const words = (s) => (s ?? "").replaceAll("_", " ");

/** Status line: what is flying, or why nothing (or only the Statcast fit) is. */
function statusLabel(state, pitch) {
  if (pitch.status === "no_tracking") return "no tracking";
  const label = pitch.status === "no_model"
    ? "Statcast fit · no model for this pitch type"
    : MODE_LABEL[state.mode];
  return shownPaths(state).some(inDirt) ? `${label} · in the dirt` : label;
}

/** Fill #hud with count, pitch, result, status and tunnel numbers for the current pitch. */
export function renderHud(state, pitch) {
  const el = document.getElementById("hud");
  const ab = currentAtBat(state);
  if (!pitch) {
    el.textContent = "Loading…";
    return;
  }
  const n = ab.pitches.length;
  const result = words(pitch.description) + (pitch.events ? ` → ${words(pitch.events)}` : "");
  const tn = pitch.tunnel;
  el.innerHTML = `
    <div class="count">${pitch.balls}-${pitch.strikes} · ${ab.outs} out · ${ab.top ? "Top" : "Bot"} ${ab.inning}</div>
    <div>${ab.game_date} · vs ${ab.stand}HB · pitch ${state.pitchIndex + 1} of ${n}</div>
    <div class="pitch">${pitch.pitch_name ?? pitch.pitch_type ?? "unknown"} · ${fmt(pitch.speed_mph, 1, " mph")}</div>
    <div>Result: ${result}</div>
    <div class="status">${statusLabel(state, pitch)}</div>
    <div class="tunnel">From the Statcast fit: VAA ${fmt(tn.vaa_deg, 1, "°")} · HAA ${fmt(tn.haa_deg, 1, "°")}
      · perceived ${fmt(tn.perceived_mph, 1, " mph")} · sep. from previous at ${state.data.meta.decision_y_ft} ft ${fmt(tn.sep_prev_in, 1, " in")}</div>`;
}
