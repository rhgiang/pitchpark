// Start-screen DOM: cards, selection, keys. Data shaping is in menu.js; main.js owns the 3D side.
import { cardModels, pitchMix, heightLabel } from "./menu.js";
import { loadPitcher, loadDelivery } from "./data.js";

// Pitch-type colours (Statcast codes); anything else is grey.
const MIX_COLOURS = { FF: "#e05a4f", SI: "#e0a24f", FC: "#c9803f", SL: "#f2c14e", ST: "#d4b84a", SV: "#b8c24a",
  CU: "#5b8def", KC: "#4f78c8", CH: "#4caf7a", FS: "#3d9a8a", Other: "#8b95a0" };
const el = (tag, cls, text) => Object.assign(document.createElement(tag), { className: cls ?? "", textContent: text ?? "" });

/**
 * Fill the cards and wait for Play ball. Resolves with the chosen slug; the menu stays open (the
 * button reads "Loading…") until the caller closes it. `pick` is called with `select(slug)` first to
 * preselect. Height, pitch mix and the delivery badge arrive lazily, one pitcher at a time, so
 * the cards show at once.
 */
export function menuPick(index, preselect) {
  const cards = cardModels(index, new Set());
  const grid = document.getElementById("cards");
  const play = document.getElementById("play-ball");
  let sel = Math.max(0, cards.findIndex((c) => c.slug === preselect));
  const nodes = cards.map((c, i) => {
    const b = el("button", "card");
    b.type = "button";
    b.setAttribute("role", "radio");
    b.append(el("b", "", c.name), Object.assign(el("div", "facts"), { textContent: `${c.hand} · ${c.nAtBats} at-bats` }));
    const badge = el("span", "badge", "Full delivery");
    badge.hidden = true;
    const mix = el("div", "mix");
    b.append(badge, mix, el("div", "mixtxt"));
    b.onclick = () => choose(i);
    return { b, badge, mix };
  });
  function choose(i, focus = false) {
    sel = i;
    nodes.forEach((n, j) => { n.b.classList.toggle("on", j === i); n.b.setAttribute("aria-checked", j === i); n.b.tabIndex = j === i ? 0 : -1; });
    if (focus) nodes[i].b.focus();
  }
  grid.replaceChildren(...nodes.map((n) => n.b));
  choose(sel);
  play.disabled = false;
  play.textContent = "Play ball ";
  play.append(Object.assign(document.createElement("kbd"), { textContent: "Enter" }));

  // Lazy extras: a sequential idle-time fetch of each pitcher's export (several MB each) for height
  // and mix; ponytail: an exported mix in index.json would make this free, add it in export_viewer.py.
  (async () => {
    for (const [i, c] of cards.entries()) {
      const d = await loadDelivery(c.slug);
      nodes[i].badge.hidden = !d;
    }
    for (const [i, c] of cards.entries()) {
      try {
        const data = await loadPitcher(c.slug);
        const mix = pitchMix(data);
        nodes[i].b.querySelector(".facts").textContent = [c.hand, heightLabel(data.pitcher.height_in), `${c.nAtBats} at-bats`].filter(Boolean).join(" · ");
        nodes[i].mix.replaceChildren(...mix.map((m) => Object.assign(el("i"), { style: `width:${m.pct}%;background:${MIX_COLOURS[m.type] ?? MIX_COLOURS.Other}` })));
        nodes[i].b.querySelector(".mixtxt").textContent = mix.slice(0, 4).map((m) => `${m.type} ${m.pct}%`).join(" · ");
      } catch (err) { console.warn(`menu info for ${c.slug} not loaded:`, err); }
      await new Promise((r) => setTimeout(r, 50)); // let input and the page breathe between big parses
    }
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
