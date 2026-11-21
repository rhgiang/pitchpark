// Sim's reveal card, at-bat strip and "call the pitches" bar. DOM only: scoring and text are in
// guess.js. Driven by main.js's events (face:thrown / face:landed / face:atbat) plus sync(state) and
// key(e). Layout, the phone pill and the hard rule (nothing over the zone or the pitcher) are in ui.css.
import { resultKind, breakLabel } from "../hud.js";
import { scoreCall, scoreType, tally, whyText, resultLine } from "./guess.js";

// Free keys for the type guess (1-6 are the view tabs; R S G N Z X C A T P F I are taken). B and S are the
// call, but only while a guess is open (key() below), so Replay's and Sim's other uses of S are unchanged.
export const TYPE_KEYS = ["KeyQ", "KeyW", "KeyE", "KeyD", "KeyH"];
const HALF_W_FT = 17 / 24; // plate half-width, for the mini zone's outline
const VAR = { b: "var(--ball)", s: "var(--strike)", f: "var(--foul)", x: "var(--inplay)" }; // ui.css result colours
const el = (tag, cls, html) => Object.assign(document.createElement(tag), { className: cls ?? "", innerHTML: html ?? "" });

/** Mini strike zone (catcher's view) as SVG: this pitch's dot, earlier pitches of the at-bat faint. */
function zoneSvg(w, h, p, prior) {
  const X = (v) => Math.min(w - 3, Math.max(3, ((v + 1.4) / 2.8) * w));
  const Z = (v) => Math.min(h - 3, Math.max(3, h - ((v - 0.8) / 3.4) * h));
  const [xl, xr, zt, zb] = [X(-HALF_W_FT), X(HALF_W_FT), Z(p.sz_top_ft), Z(p.sz_bot_ft)];
  let grid = "";
  for (let i = 1; i < 3; i++) {
    grid += `<line x1="${xl + ((xr - xl) * i) / 3}" x2="${xl + ((xr - xl) * i) / 3}" y1="${zt}" y2="${zb}"/>`
      + `<line y1="${zt + ((zb - zt) * i) / 3}" y2="${zt + ((zb - zt) * i) / 3}" x1="${xl}" x2="${xr}"/>`;
  }
  const dot = (q, r, cur) => `<circle cx="${X(q.plate_sim[0])}" cy="${Z(q.plate_sim[1])}" r="${r}" fill="${VAR[resultKind(q.description)] ?? "#fff"}"`
    + (cur ? ' stroke="#fff" stroke-width="1.5"/>' : ' opacity=".45"/>');
  return `<svg class="rv-zone" width="${w}" height="${h}" viewBox="0 0 ${w} ${h}" role="img" aria-label="Where it crossed the plate">`
    + `<g stroke="rgba(255,255,255,.28)" stroke-width=".7">${grid}</g>`
    + `<rect x="${xl}" y="${zt}" width="${xr - xl}" height="${zb - zt}" fill="none" stroke="rgba(255,255,255,.8)" stroke-width="1.2"/>`
    + prior.map((q) => dot(q, w / 20, false)).join("") + dot(p, w / 10, true) + "</svg>";
}

/**
 * `reveal` (bottom-right card) and `guess` (bottom-centre bar) are the .gp panels in index.html.
 * Returns {sync(state), key(e)}: sync after every state change; key(e) is true when a guess key was used.
 */
export function buildFaceUi({ reveal, guess }) {
  let scores = [], slug = null, face = null, stand = "R", types = [];
  let phase = "idle"; // idle | flight | landed
  let open = false, call = null, pick = null, live = null, last = null; // live: the flying pitch's pre-pitch count and Call setting

  const score = () => { const t = tally(scores); return `Calls <b>${t.calls[0]}/${t.calls[1]}</b> · Types <b>${t.types[0]}/${t.types[1]}</b>`; };
  const mark = (s) => (s === "right" ? "✓" : "✗");
  /** The at-bat strip: the count, then a chip per landed pitch (type letters on the result colour), and "?" for the one in flight. */
  const strip = (count, ab, flying) => {
    const done = ab.pitches.slice(0, ab.pitches.length - (flying ? 1 : 0));
    return `<div class="rv-strip"><span class="rv-cnt">${count}</span>` + done.map((q, i) =>
      `<span class="dot ${resultKind(q.description) ?? "b"}" title="Pitch ${i + 1}: ${q.pitch_name} ${q.speed_mph.toFixed(1)} mph · ${resultLine(q)}">${q.pitch_type}</span>`).join("")
      + (flying ? '<span class="dot q" title="In flight">?</span>' : "") + "</div>";
  };

  function render(ab) {
    document.body.classList.toggle("guessing", open);
    guess.hidden = !open;
    reveal.hidden = !(phase === "landed" || (phase === "flight" && live.callOn)); // Call off: the card appears at landing
    if (reveal.hidden) return;
    const tag = '<span class="simtag">Simulated</span>';
    if (phase === "flight") { // the type is not shown before the call: only the score and the "?" chip
      reveal.className = "gp rv flight";
      reveal.innerHTML = `<div class="rv-head">${tag}<span class="rv-score">${score()}</span></div>${strip(`${live.balls}-${live.strikes}`, ab, true)}`;
      return;
    }
    const p = ab.pitches.at(-1), s = last.scores;
    const callLine = !last.callOn ? "" : '<div class="rv-call">'
      + `Call ${s.call === "skipped" ? "<b>none</b>" : `<b>${last.call === "strike" ? "Strike" : "Ball"}</b> ${mark(s.call)}`}`
      + ` · Type ${s.type === "skipped" ? "<b>none</b>" : `<b>${last.pick}</b> ${mark(s.type)}`}`
      + (last.held ? ` <span class="rv-hint">· ${ab.event ? "N: new at-bat" : "Space: next pitch"}</span>` : "") + "</div>";
    reveal.className = `gp rv landed${reveal.classList.contains("open") ? " open" : ""}`;
    reveal.innerHTML = `<div class="rv-head">${tag}${last.callOn ? `<span class="rv-score">${score()}</span>` : `<span class="rv-meta">Pitch ${ab.pitches.length} · vs ${ab.stand}HB</span>`}</div>`
      + `<div class="rv-main"><div class="rv-info"><b class="rv-name">${p.pitch_name}</b>`
      + `<div class="rv-speed"><span class="rv-code">${p.pitch_type}</span>${p.speed_mph.toFixed(1)} <small>mph</small></div>`
      + `<div class="rv-break"><span class="mut">Break vs no spin</span><br>${breakLabel(p.tunnel)}</div></div>`
      + zoneSvg(78, 98, p, ab.pitches.slice(0, -1)) + "</div>"
      + `<div class="rv-result"><span class="dot sm ${resultKind(p.description) ?? "b"}"></span><b>${resultLine(p)}</b></div>${callLine}`
      + `<div class="rv-why"><span class="mut">Why he threw it </span>${whyText(p)}</div>`
      + strip(`${ab.balls}-${ab.strikes}`, ab, false);
  }
  // Phone: the pill expands upward in place when tapped (a user choice; it collapses again at the next pitch).
  reveal.addEventListener("click", () => reveal.classList.toggle("open"));

  function buildGuess() {
    const btn = (cls, html, onclick, data) => { const b = Object.assign(el("button", cls, html), { type: "button", onclick }); Object.assign(b.dataset, data); return b; };
    const row = el("div", "fl", "<em>Type (optional)</em>");
    types.forEach((t, i) => row.append(btn("btn pill", `${t}<kbd>${TYPE_KEYS[i]?.slice(3) ?? ""}</kbd>`, () => setPick(t), { type: t })));
    guess.replaceChildren(
      btn("btn gbtn bl", "Ball <kbd>B</kbd>", () => setCall("ball"), { call: "ball" }),
      btn("btn gbtn st", "Strike <kbd>S</kbd>", () => setCall("strike"), { call: "strike" }),
      el("span", "sep"), row);
    for (const k of guess.querySelectorAll(".pill kbd")) k.hidden = !k.textContent; // 5+ types: the rest are click-only
  }
  function paint() {
    for (const b of guess.querySelectorAll("button")) {
      const mine = b.dataset.call ? call : pick;
      b.classList.toggle("on", (b.dataset.call ?? b.dataset.type) === mine);
      b.classList.toggle("locked", mine !== null); // a call is final once made
    }
  }
  const setCall = (c) => { if (open && call === null) { call = c; paint(); } };
  const setPick = (t) => { if (open && pick === null) { pick = t; paint(); } };

  function reset(state) { // leaving Sim, or a different pitcher: the score starts over (new at-bats keep it)
    if (state.kind !== "face") { slug = null; scores = []; phase = "idle"; open = false; render(null); return; }
    if (slug !== state.slug) { slug = state.slug; scores = []; phase = "idle"; open = false; reveal.hidden = guess.hidden = true; }
    face = state.face.data; stand = state.face.ab.stand; types = Object.keys(face.bank);
  }

  document.addEventListener("face:thrown", (e) => {
    const { state, ab, pitch } = e.detail;
    reset(state);
    phase = "flight"; call = pick = null; open = state.face.opts.call;
    live = { balls: pitch.balls, strikes: pitch.strikes, callOn: open };
    reveal.classList.remove("open");
    if (open) buildGuess();
    render(ab);
  });
  document.addEventListener("face:landed", (e) => {
    const { state, ab, pitch } = e.detail;
    if (phase !== "flight") return;
    const callOn = state.face.opts.call;
    // No call scores as skipped (not counted). The bar closes here, so a guess after landing cannot count.
    const s = callOn ? { call: scoreCall(call, pitch, { face, stand }), type: scoreType(pick, pitch) } : { call: "skipped", type: "skipped" };
    if (callOn) scores.push(s);
    // Call on and nothing guessed or dismissed yet: hold the auto-advance so the card can be read.
    const held = callOn && s.call === "skipped" && s.type === "skipped";
    if (held) e.preventDefault();
    last = { scores: s, call, pick, callOn, held };
    phase = "landed"; open = false;
    render(ab);
  });
  document.addEventListener("face:atbat", (e) => {
    reset(e.detail.state);
    phase = "idle"; open = false;
    render(e.detail.ab);
  });

  return {
    sync(state) { if (state.kind !== "face" || slug !== state.slug) reset(state); },
    key(e) {
      if (!open) return false;
      const i = TYPE_KEYS.indexOf(e.code);
      if (e.code === "KeyB") setCall("ball");
      else if (e.code === "KeyS") setCall("strike");
      else if (i >= 0 && types[i]) setPick(types[i]);
      else return false;
      return true;
    },
  };
}
