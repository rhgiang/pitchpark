import { currentAtBat, shownPaths, landed, spinFactor, revealed, MODES, SPEEDS, VIEWS } from "./state.js";
import { inDirt } from "./ball.js";

// The broadcast interface's text: scorebug, status line, pitch strip, "the look" card, and which
// buttons are lit. Spoiler rule: results (result text, at-bat event, "in the dirt", strip colours,
// break vs no spin) only once that pitch has landed or been passed.

const MODE_LABEL = { real: "Statcast fit", sim: "M1 + M3", ghost: "Statcast fit + M1 + M3 ghost" };
const VIEW_LABEL = {
  umpire: "Umpire's view", batter: "Batter's eye", side: "Side view (first-base side)",
  ride: "Riding with the ball", free: "Free camera: drag to orbit, wheel to zoom", tv: "TV view (centre-field camera)",
};

const fmt = (v, digits, unit = "") => (v == null ? "–" : `${v.toFixed(digits)}${unit}`);
const words = (s) => (s ?? "").replaceAll("_", " ");
const cap = (s) => s.charAt(0).toUpperCase() + s.slice(1);

/**
 * Result class of a Statcast `description` for the pitch strip: "b" ball, "s" strike, "f" foul,
 * "x" in play; null when unknown. Foul tips are strikes (caught); pitchouts are balls unless
 * swung at.
 */
export function resultKind(desc) {
  if (!desc) return null;
  if (desc === "hit_into_play" || desc.startsWith("hit_into_play")) return "x";
  if (desc.startsWith("foul") && desc !== "foul_tip") return "f";
  if (/strike|foul_tip|missed_bunt|swinging/.test(desc)) return "s";
  return "b"; // ball, blocked_ball, intent_ball, pitchout, hit_by_pitch
}

/** Status line: what is flying, or why nothing (or only the Statcast fit) is, and the view. */
function statusLabel(state, pitch) {
  let label;
  if (pitch.status === "no_tracking") label = `<span class="warn">No tracking</span> for this pitch`;
  else if (pitch.status === "no_model") label = `Statcast fit · <span class="warn">no model for this pitch type</span>`;
  else label = MODE_LABEL[state.mode];
  // Only after landing: "in the dirt" before release would give the pitch away.
  if (landed(state) && shownPaths(state).some(inDirt)) label += " · in the dirt";
  return `${label} · ${VIEW_LABEL[state.view]}`;
}

/** Spin note: whether the on-screen rotation is slowed (it would alias at 60 Hz otherwise). */
function spinNote(state, pitch) {
  if (pitch.spin_rpm == null || pitch.spin_axis_deg == null) return "Spin unknown.";
  const rps = pitch.spin_rpm / 60;
  const f = spinFactor(state.speed);
  const shown = state.speed >= 1
    ? `slowed to 1/${Math.round(1 / f)} of true (${(rps * f).toFixed(1)} rev/s shown)`
    : `at the true rate in flight time (${state.speed}× playback)`;
  return `Ball rotation ${shown}; the blue rod is the spin axis.`;
}

/** Break vs no spin as "12.3 in ↓ · 4.1 in ←" (from the catcher's side: + x is right, + z up). */
function breakLabel(tn) {
  if (tn.break_x_in == null) return "–";
  const z = `${Math.abs(tn.break_z_in).toFixed(1)} in ${tn.break_z_in >= 0 ? "↑" : "↓"}`;
  const x = `${Math.abs(tn.break_x_in).toFixed(1)} in ${tn.break_x_in >= 0 ? "→" : "←"}`;
  return `${z} · ${x}`;
}

const $ = (id) => document.getElementById(id);

/** Refresh the whole interface for `state` (cheap: called on every action, not every frame). */
export function renderHud(state, pitch) {
  for (const m of MODES) {
    const b = $(`mode-${m}`);
    b.disabled = m !== "real" && !state.modeAllowed;
    b.classList.toggle("on", state.mode === m);
  }
  for (const v of VIEWS) $(`view-${v}`).classList.toggle("on", state.view === v);
  for (const v of SPEEDS) $(`speed-${v}`).classList.toggle("on", state.speed === v);
  $("nospin").classList.toggle("on", state.showNoSpin);
  $("nospin").disabled = !pitch?.nospin;
  if (!pitch) return;

  const ab = currentAtBat(state);
  const done = landed(state);
  // Scorebug: the count, outs and runners before this pitch; pitch type and speed are known at
  // release, so they show at once.
  $("bug-count").textContent = `${pitch.balls}-${pitch.strikes}`;
  const surname = state.data.pitcher.name.split(",")[0];
  $("bug-pitch").textContent = `${surname} · ${pitch.pitch_name ?? pitch.pitch_type ?? "Unknown"} ${fmt(pitch.speed_mph, 1, " mph")}`;
  $("bug-inning").textContent = `${ab.top ? "Top" : "Bot"} ${ab.inning} · ${ab.outs} out · vs ${ab.stand}HB`;
  (pitch.bases ?? [false, false, false]).forEach((on, i) => $(`base-${i + 1}`).classList.toggle("on", on));
  $("bug-status").innerHTML = statusLabel(state, pitch);

  // Pitch strip: number until revealed, then the result letter on its colour.
  const dots = $("dots").children;
  ab.pitches.forEach((p, i) => {
    const kind = revealed(state, i) ? resultKind(p.description) : null;
    dots[i].className = `dot${kind ? ` ${kind}` : ""}${i === state.pitchIndex ? " cur" : ""}${i > state.pitchIndex ? " ahead" : ""}`;
    dots[i].textContent = kind ? kind.toUpperCase() : String(i + 1);
    // Pitch type only for pitches already thrown (the current one is known at release).
    const name = i <= state.pitchIndex || revealed(state, i) ? `: ${p.pitch_name ?? p.pitch_type ?? "?"}` : "";
    dots[i].title = `Pitch ${i + 1}${name}${kind ? ` · ${words(p.description)}` : ""}`;
  });
  const last = state.pitchIndex === ab.pitches.length - 1;
  $("result").innerHTML = !done ? "" : cap(words(pitch.description))
    + (last && ab.event ? ` <span class="ev">→ ${cap(words(ab.event))}</span>` : "");

  const tn = pitch.tunnel;
  $("look-body").innerHTML = `<dl>
    <dt>VAA · HAA</dt><dd>${fmt(tn.vaa_deg, 1, "°")} · ${fmt(tn.haa_deg, 1, "°")}</dd>
    <dt>Perceived</dt><dd>${fmt(tn.perceived_mph, 1, " mph")}</dd>
    <dt>Break vs no spin</dt><dd class="accent">${done ? breakLabel(tn) : "after the pitch"}</dd>
    <dt>Sep. from prev. at ${state.data.meta.decision_y_ft} ft</dt><dd>${fmt(tn.sep_prev_in, 1, " in")}</dd>
    <dt>Spin · axis</dt><dd>${fmt(pitch.spin_rpm, 0, " rpm")} · ${fmt(pitch.spin_axis_deg, 0, "°")}</dd>
    <dt>Extension</dt><dd>${fmt(pitch.extension_ft, 1, " ft")}</dd>
  </dl>
  <div class="note">Angles and perceived speed from the Statcast fit. Break: Statcast fit minus the
  no-spin path at the plate. ${spinNote(state, pitch)}</div>`;
}

/** Build one clickable dot per pitch of the current at-bat (on at-bat change). */
export function buildStrip(state, onPick) {
  $("dots").replaceChildren(...currentAtBat(state).pitches.map((_, i) => {
    const b = document.createElement("button");
    b.onclick = () => onPick(i);
    return b;
  }));
}
