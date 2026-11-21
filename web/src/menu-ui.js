// Start-screen DOM: cards, selection, keys. Data shaping is in menu.js; main.js owns the 3D side.
import { cardModels } from "./menu.js";
import { loadDelivery, prefetchPitcher } from "./data.js";
import { PACES } from "./state.js";
import { PLAY_LABEL } from "./face/labels.js";

// Pitch-type colours (Statcast codes); anything else is grey.
const MIX_COLOURS = { FF: "#e05a4f", SI: "#e0a24f", FC: "#c9803f", SL: "#f2c14e", ST: "#d4b84a", SV: "#b8c24a",
  CU: "#5b8def", KC: "#4f78c8", CH: "#4caf7a", FS: "#3d9a8a", Other: "#8b95a0" };
const el = (tag, cls, text) => Object.assign(document.createElement(tag), { className: cls ?? "", textContent: text ?? "" });

// ---- Sim controls (start row and in-game bar share these) --------------------------------------
export const FACE_DEFAULTS = { stand: "R", batter: "avg", call: false, startCount: "0-0", paceS: PACES.real };
const STAND = [["L", "L"], ["R", "R"]], BATTER = [["none", "None"], ["avg", "Average hitter"]];
const COUNTS = ["0-0", "1-1", "0-2", "3-2"].map((c) => [c, c]);
const PACE = [["fast", "Fast"], ["real", "Real"], ["slow", "Slow"]];
export const paceName = (s) => Object.entries(PACES).find(([, v]) => v === s)?.[0] ?? "real";

/** Labelled segmented switch: {el, set(value)}. `items`: [[value, text]]; onPick(value) on a click. */
export function segment(label, items, value, onPick) {
  const wrap = el("div", "fl"), seg = el("span", "seg");
  wrap.append(el("em", "", label), seg);
  const btns = items.map(([v, text]) => {
    const b = el("button", "", text);
    b.type = "button";
    b.onclick = () => { set(v); onPick(v); };
    seg.append(b);
    return [v, b];
  });
  function set(v) { for (const [k, b] of btns) { b.classList.toggle("on", k === v); b.setAttribute("aria-pressed", String(k === v)); } }
  set(value);
  return { el: wrap, set };
}

/** Labelled on/off switch: {el, set(on)}. */
export function toggle(label, on, onPick) {
  const wrap = el("div", "fl"), b = el("button", "btn toggle sw-btn");
  b.type = "button";
  b.append(el("span", "sw"), el("small"));
  wrap.append(el("em", "", label), b);
  const set = (v) => { b.classList.toggle("on", v); b.setAttribute("aria-pressed", String(v)); b.lastChild.textContent = v ? "on" : "off"; on = v; };
  b.onclick = () => { set(!on); onPick(on); };
  set(on);
  return { el: wrap, set };
}

/**
 * Slim in-game bar (bottom-centre) into `host`: New at-bat, Next pitch, Pace and the Call toggle. Returns
 * update(state): the pace and call switches follow face.opts, Next pitch is off in flight and after the
 * last pitch of an at-bat (where New at-bat becomes the primary button).
 */
export function buildFaceBar(host, { onNew, onNext, onPace, onCall }) {
  const nu = el("button", "btn", "↻ New at-bat "), next = el("button", "btn big", "Next pitch ");
  nu.type = next.type = "button";
  nu.append(Object.assign(el("kbd"), { textContent: "N" }));
  next.append(Object.assign(el("kbd"), { textContent: "Space" }));
  nu.onclick = onNew;
  next.onclick = onNext;
  const pace = segment("Pace", PACE, "real", (v) => onPace(PACES[v]));
  const call = toggle("Call", false, onCall);
  host.replaceChildren(nu, next, el("span", "sep"), pace.el, el("span", "sep"), call.el);
  return (opts, canThrow, over) => {
    pace.set(paceName(opts.paceS));
    call.set(opts.call);
    next.disabled = !canThrow;
    next.classList.toggle("big", !over);
    nu.classList.toggle("big", over);
  };
}

/**
 * The choices made on the start screen survive reopening it ("Change pitcher") and also feed the F key:
 * {face: Sim selected, opts: face options}. A deep link (?face=1...) seeds them via `initial`.
 */
const prefs = { face: false, opts: { ...FACE_DEFAULTS } };
export const facePrefs = () => ({ ...prefs.opts });

/**
 * Fill the cards and wait for Play ball. Resolves with the chosen slug; the menu stays open (the
 * button reads "Loading…") until the caller closes it. `pick` is called with `select(slug)` first to
 * preselect. Height and pitch mix come from index.json; the delivery badges arrive lazily.
 * Resolves with {slug, face}: face is the Sim options if that mode is selected, else null (Replay).
 * `initial` ({face, opts}) preselects the mode and options (a deep link).
 */
export function menuPick(index, preselect, initial) {
  if (initial) { prefs.face = initial.face; prefs.opts = { ...prefs.opts, ...initial.opts }; }
  const cards = cardModels(index, new Set());
  const grid = document.getElementById("cards");
  const play = document.getElementById("play-ball");
  let sel = Math.max(0, cards.findIndex((c) => c.slug === preselect));
  const nodes = cards.map((c, i) => {
    const b = el("button", "card");
    b.type = "button";
    b.dataset.slug = c.slug; // guide-ui.js puts the loading bar on this card
    b.style.setProperty("--i", i); // entrance stagger (ui.css)
    b.setAttribute("role", "radio");
    b.append(el("b", "", c.name), Object.assign(el("div", "facts"), { textContent: [c.hand, c.height, `${c.nAtBats} at-bats`].filter(Boolean).join(" · ") }));
    const badge = el("span", "badge", "Full delivery");
    badge.hidden = true;
    const mix = el("div", "mix");
    b.append(badge, mix, Object.assign(el("div", "mixtxt"), { textContent: c.mix.slice(0, 4).map((m) => `${m.type} ${m.pct}%`).join(" · ") }));
    mix.replaceChildren(...c.mix.map((m) => Object.assign(el("i"), { style: `width:${m.pct}%;background:${MIX_COLOURS[m.type] ?? MIX_COLOURS.Other}` })));
    b.onclick = () => choose(i);
    // Start the download as soon as someone shows interest; the real load later reuses it.
    for (const ev of ["pointerenter", "focus"]) b.addEventListener(ev, () => prefetchPitcher(c.slug));
    return { b, badge, mix };
  });
  const surname = (i) => index.find((p) => p.slug === cards[i].slug)?.name.split(",")[0] ?? cards[i].name;
  function updatePlay() {
    if (play.disabled) return;
    play.textContent = prefs.face ? `${PLAY_LABEL(surname(sel))} ` : "Play ball ";
    play.append(Object.assign(document.createElement("kbd"), { textContent: "Enter" }));
  }
  function choose(i, focus = false) {
    sel = i;
    updatePlay();
    prefetchPitcher(cards[i].slug);
    nodes.forEach((n, j) => { n.b.classList.toggle("on", j === i); n.b.setAttribute("aria-checked", j === i); n.b.tabIndex = j === i ? 0 : -1; });
    if (focus) nodes[i].b.focus();
  }
  grid.replaceChildren(...nodes.map((n) => n.b));
  choose(sel);
  play.disabled = false;
  updatePlay();

  // Replay | Sim, and (Sim only) the options row between the cards and the play button.
  const modeBtns = [...document.querySelectorAll("#mode-seg button")], row = document.getElementById("face-opts");
  const o = prefs.opts;
  const ctl = {
    stand: segment("Side", STAND, o.stand, (v) => { o.stand = v; }),
    batter: segment("Batter", BATTER, o.batter, (v) => { o.batter = v; }),
    call: toggle("Call the pitches", o.call, (v) => { o.call = v; }),
    count: segment("Count", COUNTS, o.startCount, (v) => { o.startCount = v; }),
    pace: segment("Pace", PACE, paceName(o.paceS), (v) => { o.paceS = PACES[v]; }),
  };
  // "New at-bat" here just starts one (there is nothing to restart yet): same as the play button.
  const fresh = el("button", "btn", "↻ New at-bat ");
  fresh.type = "button";
  fresh.append(Object.assign(el("kbd"), { textContent: "N" }));
  row.replaceChildren(...Object.values(ctl).map((c) => c.el), fresh);
  const setFace = (on) => {
    prefs.face = on;
    modeBtns.forEach((b) => { b.classList.toggle("on", (b.dataset.mode === "face") === on); b.setAttribute("aria-checked", String((b.dataset.mode === "face") === on)); });
    row.hidden = !on;
    document.body.classList.toggle("face-menu", on); // phone: the row becomes a docked sheet (ui.css)
    updatePlay();
  };
  modeBtns.forEach((b) => { b.onclick = () => setFace(b.dataset.mode === "face"); });
  setFace(prefs.face);

  // The delivery badge needs a (small) delivery file per pitcher; the cache makes a later pick free.
  (async () => {
    for (const [i, c] of cards.entries()) nodes[i].badge.hidden = !(await loadDelivery(c.slug));
  })();

  return new Promise((resolve) => {
    const go = () => {
      play.disabled = true;
      play.textContent = "Loading…";
      window.removeEventListener("keydown", onKey);
      document.body.classList.remove("face-menu");
      resolve({ slug: cards[sel].slug, face: prefs.face ? { ...prefs.opts } : null });
    };
    function onKey(e) {
      if (e.ctrlKey || e.metaKey || e.altKey) return;
      const step = { ArrowRight: 1, ArrowDown: 1, ArrowLeft: -1, ArrowUp: -1 }[e.key];
      if (step) { e.preventDefault(); choose((sel + step + cards.length) % cards.length, true); }
      else if (e.key === "Enter") { e.preventDefault(); go(); }
      else if (e.code === "KeyF") { e.preventDefault(); setFace(!prefs.face); }
      else if (e.code === "KeyN" && prefs.face) { e.preventDefault(); go(); }
    }
    window.addEventListener("keydown", onKey);
    play.onclick = go;
    fresh.onclick = go;
  });
}
