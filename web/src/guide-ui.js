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
 * Glass message card. Options:
 *   title, text, tone ("error": red edge)
 *   buttons: [{label, big?, keep?, onClick}]  a click closes the card first, unless keep (e.g. "How it works")
 *   choices: [{label, onClick}]  a row of smaller buttons above `buttons` (the compare page's pitcher list)
 *   icon (large glyph above the title), spinner (inline), layout "stack" (centred column) | default inline row
 *   pos "corner" | "bottom": a placement instead of the screen centre
 *   modal (default true): traps Tab and swallows keys so Space does not throw a pitch behind it, and
 *     moves focus to the primary button (back to where it was on close). modal:false leaves the page usable;
 *     `focus: true` still moves focus to the primary button.
 *   dismissible: Esc closes it.
 * Returns {close, card, buttons}; close() is safe to call twice.
 */
export function showCard({ title, text, tone, icon, spinner, layout, pos, modal = true, focus = modal, dismissible = false, choices = [], buttons = [] }) {
  const prev = document.activeElement;
  const card = el("div", `gp msg${tone === "error" ? " err" : ""}${layout === "stack" ? " stack" : ""}${pos ? ` ${pos}` : ""}`);
  card.setAttribute("role", modal ? "alertdialog" : "dialog");
  card.setAttribute("aria-modal", String(modal));
  card.setAttribute("aria-label", title);
  const mk = (b, big) => {
    const n = el("button", `btn${big ? " big" : ""}`, b.label);
    n.type = "button";
    n.onclick = () => { if (!b.keep) close(); b.onClick?.(); };
    return n;
  };
  const btns = buttons.map((b) => mk(b, b.big));
  const picks = choices.map((c) => mk(c, false));
  const row = el("div", "msg-row");
  row.append(...btns);
  const t = el("div", "msg-text");
  t.append(el("b", "", title), el("span", "", text ?? ""));
  const glyph = icon ?? (layout === "stack" || spinner ? null : tone === "error" ? "!" : "i");
  card.append(...[
    glyph && el("span", `msg-icon${icon ? " big" : ""}`, glyph), spinner && el("span", "spin"), t,
    picks.length && el("div", "msg-choices"), row,
  ].filter(Boolean));
  if (picks.length) card.querySelector(".msg-choices").append(...picks);
  document.body.append(card);
  if (focus) (btns.find((n) => n.classList.contains("big")) ?? btns[0] ?? picks[0])?.focus();
  const onEsc = (e) => { if (dismissible && e.key === "Escape") close(); };
  let untrap = null;
  if (modal) untrap = trapKeys(card, onEsc); else card.addEventListener("keydown", onEsc);
  let open = true;
  function close() {
    if (!open) return;
    open = false;
    untrap?.();
    card.remove();
    if (focus && prev?.isConnected) prev.focus?.();
  }
  return { close, card, buttons: btns };
}

/** showCard whose buttons carry a `value`: resolves with the clicked button's value. */
export const askCard = (opts) => new Promise((resolve) => showCard({
  ...opts, buttons: opts.buttons.map((b) => ({ ...b, onClick: () => resolve(b.value) })),
}));

// ---- toast ---------------------------------------------------------------------------------------
/** A key cap for toast text: toast("Press ", kbd("?"), " anytime"). */
export const kbd = (key) => el("kbd", "", key);
/** Small glass pill, announced politely, gone after ~3 s. `parts` are strings and/or nodes. */
export function toast(...parts) {
  let live = document.getElementById("toasts");
  if (!live) { // one polite live region, created before its first message so screen readers pick it up
    live = Object.assign(el("div"), { id: "toasts" });
    live.setAttribute("aria-live", "polite");
    document.body.append(live);
  }
  const n = el("div", "gp toast");
  const msg = el("span");
  msg.append(...parts);
  n.append(el("span", "ok", "✓"), msg);
  live.replaceChildren(n);
  setTimeout(() => n.classList.add("out"), 2600);
  setTimeout(() => n.remove(), 3100);
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
      const choice = await askCard({
        tone: "error", title: errorText(name), text: "Check your connection and try again.",
        buttons: [{ label: "Retry", big: true, value: "retry" }, { label: "Back to menu", value: "menu" }],
      });
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
    row.style.setProperty("--i", i); // cascade stagger (cards.css)
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
  // Done (Next on the last tip) is finishing; Skip / Esc / clicking away are not, so only Done gets the toast.
  function advance() {
    const to = nextStep(i, TOUR_STEPS.length);
    if (to === null) { close(); toast("You're all set. Press ", kbd("?"), " anytime for controls and the tour."); } else go(to);
  }
  function close() { untrap(); window.removeEventListener("resize", place); root.remove(); tourOpen = false; prev?.isConnected && prev.focus?.(); }
  const untrap = trapKeys(card, (e) => {
    if (e.key === "Escape") close();
    else if (e.key === "ArrowRight") advance();
    else if (e.key === "ArrowLeft") go(prevStep(i));
  });
  skip.onclick = close;
  next.onclick = advance;
  shade.onclick = close;
  window.addEventListener("resize", place);
  show();
}
