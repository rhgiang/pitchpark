// Start-screen DOM: cards, selection, keys. Data shaping is in menu.js; main.js owns the 3D side.
import { cardModels } from "./menu.js";
import { loadDelivery, prefetchPitcher } from "./data.js";

// Pitch-type colours (Statcast codes); anything else is grey.
const MIX_COLOURS = { FF: "#e05a4f", SI: "#e0a24f", FC: "#c9803f", SL: "#f2c14e", ST: "#d4b84a", SV: "#b8c24a",
  CU: "#5b8def", KC: "#4f78c8", CH: "#4caf7a", FS: "#3d9a8a", Other: "#8b95a0" };
const el = (tag, cls, text) => Object.assign(document.createElement(tag), { className: cls ?? "", textContent: text ?? "" });

/**
 * Fill the cards and wait for Play ball. Resolves with the chosen slug; the menu stays open (the
 * button reads "Loading…") until the caller closes it. `pick` is called with `select(slug)` first to
 * preselect. Height and pitch mix come from index.json; the delivery badges arrive lazily.
 */
export function menuPick(index, preselect) {
  const cards = cardModels(index, new Set());
  const grid = document.getElementById("cards");
  const play = document.getElementById("play-ball");
  let sel = Math.max(0, cards.findIndex((c) => c.slug === preselect));
  const nodes = cards.map((c, i) => {
    const b = el("button", "card");
    b.type = "button";
    b.dataset.slug = c.slug; // guide-ui.js puts the loading bar on this card
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
  function choose(i, focus = false) {
    sel = i;
    prefetchPitcher(cards[i].slug);
    nodes.forEach((n, j) => { n.b.classList.toggle("on", j === i); n.b.setAttribute("aria-checked", j === i); n.b.tabIndex = j === i ? 0 : -1; });
    if (focus) nodes[i].b.focus();
  }
  grid.replaceChildren(...nodes.map((n) => n.b));
  choose(sel);
  play.disabled = false;
  play.textContent = "Play ball ";
  play.append(Object.assign(document.createElement("kbd"), { textContent: "Enter" }));

  // The delivery badge needs a (small) delivery file per pitcher; the cache makes a later pick free.
  (async () => {
    for (const [i, c] of cards.entries()) nodes[i].badge.hidden = !(await loadDelivery(c.slug));
  })();

  return new Promise((resolve) => {
    const go = () => {
      play.disabled = true;
      play.textContent = "Loading…";
      window.removeEventListener("keydown", onKey);
      resolve(cards[sel].slug);
    };
    function onKey(e) {
      if (e.ctrlKey || e.metaKey || e.altKey) return;
      const step = { ArrowRight: 1, ArrowDown: 1, ArrowLeft: -1, ArrowUp: -1 }[e.key];
      if (step) { e.preventDefault(); choose((sel + step + cards.length) % cards.length, true); }
      else if (e.key === "Enter") { e.preventDefault(); go(); }
    }
    window.addEventListener("keydown", onKey);
    play.onclick = go;
  });
}
