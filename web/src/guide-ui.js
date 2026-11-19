// DOM for the first-visit tour, the "How it works" panel and loading feedback. Data and pure logic: guide.js.
// These panels use .gp (ui.css), not .glass: body.menu / body.intro hide every .glass, and "How it
// works" must open from the start menu.
import { TOUR_STEPS, markTourSeen, resolveTarget, placeCard, nextStep, prevStep, HOW, loadingText, errorText } from "./guide.js";

const el = (tag, cls, text) => Object.assign(document.createElement(tag), { className: cls ?? "", textContent: text ?? "" });
const FOCUSABLE = "button, a[href], [tabindex]:not([tabindex='-1'])";

/** Keep Tab inside `root`, and stop every other key from reaching the app underneath (Space would throw a pitch). */
function trapKeys(root, onKey) {
  const handler = (e) => {
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    if (e.key === "Tab") {
      const f = [...root.querySelectorAll(FOCUSABLE)].filter((n) => !n.disabled && n.offsetParent !== null);
      if (!f.length) return e.preventDefault();
      const i = f.indexOf(document.activeElement);
      const to = e.shiftKey ? f[(i <= 0 ? f.length : i) - 1] : f[(i + 1) % f.length];
      e.preventDefault();
      return to.focus();
    }
    // Space/Enter on a focused button must still click it: let those through untouched.
    if ((e.key === " " || e.key === "Enter") && document.activeElement?.closest?.("button, a")) return e.stopImmediatePropagation();
    e.stopImmediatePropagation();
    onKey(e);
  };
  window.addEventListener("keydown", handler, true);
  return () => window.removeEventListener("keydown", handler, true);
}

// ---- generic message card: title, a sentence, one or two buttons ------------------------------------
/**
 * Centred glass card. buttons: [{label, big?, onClick}]; a click closes the card first. Returns {close}.
 * Reused for the load-error panel; Part 2 error pages can call it too.
 */
export function showCard({ title, text, tone, buttons = [] }) {
  const prev = document.activeElement;
  const card = el("div", `gp msg${tone === "error" ? " err" : ""}`);
  card.setAttribute("role", "alertdialog");
  card.setAttribute("aria-label", title);
  const row = el("div", "msg-row");
  const btns = buttons.map((b) => {
    const n = el("button", `btn${b.big ? " big" : ""}`, b.label);
    n.type = "button";
    n.onclick = () => { close(); b.onClick?.(); };
    return n;
  });
  row.append(...btns);
  card.append(el("span", "msg-icon", tone === "error" ? "!" : "i"), (() => { const t = el("div", "msg-text"); t.append(el("b", "", title), el("span", "", text)); return t; })(), row);
  document.body.append(card);
  btns.find((n) => n.classList.contains("big"))?.focus() ?? btns[0]?.focus();
  const untrap = trapKeys(card, () => {});
  function close() { untrap(); card.remove(); prev?.isConnected && prev.focus?.(); }
  return { close };
}

// ---- loading feedback ----------------------------------------------------------------------------
/** Run `fn()` with a progress bar on the chosen menu card and a "Loading <name>'s delivery…" chip. */
export async function withLoading(slug, name, fn) {
  const card = document.querySelector(`#cards [data-slug="${slug}"]`);
  const bar = el("div", "loadbar"); bar.append(el("i"));
  card?.append(bar);
  const chip = el("div", "gp chip");
  chip.setAttribute("role", "status");
  chip.append(el("span", "spin"), el("span", "", loadingText(name)));
  document.body.append(chip);
  try { return await fn(); } finally { bar.remove(); chip.remove(); }
}

/**
 * withLoading, then on failure an error card. Resolves with fn's result, or `null` when the user
 * chose "Back to menu" (the caller reopens the menu).
 */
export async function loadOrAsk(slug, name, fn) {
  for (;;) {
    try { return await withLoading(slug, name, fn); } catch (err) {
      console.error(`loading ${slug} failed:`, err);
      const choice = await new Promise((resolve) => showCard({
        tone: "error", title: errorText(name), text: "Check your connection and try again.",
        buttons: [{ label: "Retry", big: true, onClick: () => resolve("retry") }, { label: "Back to menu", onClick: () => resolve("menu") }],
      }));
      if (choice === "menu") return null;
    }
  }
}

// ---- "How it works" ------------------------------------------------------------------------------
let howOpen = null;
export function openHow() {
  if (howOpen) return;
  const prev = document.activeElement;
  const back = el("div", "gp-back");
  const panel = el("div", "gp how");
  panel.setAttribute("role", "dialog");
  panel.setAttribute("aria-modal", "true");
  panel.setAttribute("aria-labelledby", "how-title");
  const x = el("button", "how-x"); x.type = "button"; x.setAttribute("aria-label", "Close");
  x.append("✕ ", Object.assign(el("kbd"), { textContent: "Esc" }));
  const h = el("h3", "", HOW.title); h.id = "how-title";
  const pipe = el("div", "pipe");
  HOW.steps.forEach((s, i) => {
    const row = el("div", "st"), t = el("div", "t"), num = el("div", "num", s.n);
    t.append(el("b", "", s.t), el("span", "", s.d));
    num.append(el("small", "", s.c));
    row.append(el("div", "n", String(i + 1)), t, num);
    pipe.append(row);
  });
  panel.append(x, h, el("div", "sub", HOW.sub), pipe, el("div", "disc", HOW.disclaimer));
  back.append(panel);
  document.body.append(back);
  const untrap = trapKeys(panel, (e) => { if (e.key === "Escape") close(); });
  function close() { untrap(); back.remove(); howOpen = null; prev?.isConnected && prev.focus?.(); }
  x.onclick = close;
  back.onclick = (e) => { if (e.target === back) close(); };
  howOpen = { close };
  x.focus();
}

// ---- first-visit tour ----------------------------------------------------------------------------
let tourOpen = false;
/** Spotlight tour over TOUR_STEPS. Marks itself seen when it opens (a reload mid-tour will not nag again). */
export function startTour() {
  if (tourOpen) return;
  tourOpen = true;
  markTourSeen();
  const prev = document.activeElement;
  const root = el("div", "tour");
  const shade = el("div", "tour-shade"), ring = el("div", "tour-ring");
  const card = el("div", "gp tip");
  card.setAttribute("role", "dialog");
  card.setAttribute("aria-label", "Quick tour");
  const step = el("div", "step"), title = el("b"), body = el("p"), pips = el("div", "pips");
  const skip = el("button", "btn", "Skip"), next = el("button", "btn big");
  skip.type = next.type = "button";
  const row = el("div", "row"), acts = el("div", "acts");
  acts.append(skip, next);
  row.append(pips, acts);
  card.append(step, title, body, row);
  root.append(shade, ring, card);
  document.body.append(root);
  let i = 0;

  const rectOf = (sel) => { const n = document.querySelector(sel); return n && (n.parentElement?.classList.contains("seg") ? n.parentElement : n).getBoundingClientRect(); };
  function show() {
    const s = TOUR_STEPS[i];
    step.textContent = `Tip ${i + 1} of ${TOUR_STEPS.length}`;
    title.textContent = s.title;
    body.innerHTML = s.html;
    pips.replaceChildren(...TOUR_STEPS.map((_, j) => el("i", j === i ? "on" : "")));
    next.textContent = i === TOUR_STEPS.length - 1 ? "Done" : "Next";
    place();
    next.focus();
  }
  function place() {
    const s = TOUR_STEPS[i];
    const r = resolveTarget(s.target, rectOf);
    root.classList.toggle("noring", !r); // hidden target (phone width): dim everything, card centred
    if (r) Object.assign(ring.style, { left: `${r.left}px`, top: `${r.top}px`, width: `${r.width}px`, height: `${r.height}px` });
    const c = card.getBoundingClientRect();
    const p = placeCard(r, { width: c.width, height: c.height }, window.innerWidth, window.innerHeight);
    Object.assign(card.style, { left: `${p.left}px`, top: `${p.top}px` });
  }
  function go(to) { if (to === null) return close(); i = to; show(); }
  function close() { untrap(); window.removeEventListener("resize", place); root.remove(); tourOpen = false; prev?.isConnected && prev.focus?.(); }
  const untrap = trapKeys(card, (e) => {
    if (e.key === "Escape") close();
    else if (e.key === "ArrowRight") go(nextStep(i, TOUR_STEPS.length));
    else if (e.key === "ArrowLeft") go(prevStep(i));
  });
  skip.onclick = close;
  next.onclick = () => go(nextStep(i, TOUR_STEPS.length));
  shade.onclick = close;
  window.addEventListener("resize", place);
  show();
}
