// The "something is not right" cards of the viewer and the compare page (Part 2 of the UX work).
// Each one is a showCard (guide-ui.js); the pure rules (slow detector, flags, lists) are in health.js.
import { showCard, askCard, openHow } from "./guide-ui.js";
import { loadIndex } from "./data.js";
import { compareChoices, getFlag, setFlag, PHONE_KEY, isPhone } from "./health.js";
import { displayName } from "./menu.js";

let fatalShown = false;

/** A: no WebGL. The card stays up; "How it works" opens over it and needs no 3D. */
export function showNoWebGL() {
  fatalShown = true;
  showCard({
    icon: "⚾", layout: "stack", modal: false, focus: true,
    title: "Pitch Park needs 3D graphics",
    text: "Your browser has WebGL turned off or doesn't support it. Try a recent Chrome, Edge, Firefox or Safari, or switch on hardware acceleration in your browser settings.",
    buttons: [{ label: "How it works (no 3D needed)", big: true, keep: true, onClick: openHow }],
  });
}

/** Last resort for anything uncaught: log it, and show one friendly card per page load. */
export function installGlobalHandlers() {
  const report = (err) => {
    console.error("uncaught:", err);
    if (fatalShown) return;
    fatalShown = true;
    showCard({
      layout: "stack", tone: "error",
      title: "Something went wrong", text: "Pitch Park hit an unexpected snag. Reloading usually fixes it.",
      buttons: [{ label: "Reload", big: true, onClick: () => location.reload() }, { label: "Keep going" }],
    });
  };
  window.addEventListener("error", (e) => {
    // A failed <img>/<script> has no message; ResizeObserver's loop notice is harmless noise.
    if (!e.message || /ResizeObserver loop/.test(e.message)) return;
    report(e.error ?? e.message);
  });
  window.addEventListener("unhandledrejection", (e) => report(e.reason));
}

/** B: the pitcher index. Resolves with it; a failure (or an empty list) asks "Try again" until it loads. */
export async function loadIndexOrAsk(load = loadIndex) {
  for (;;) {
    try {
      const index = await load();
      if (!Array.isArray(index) || !index.length) throw new Error("pitcher index is empty");
      return index;
    } catch (err) {
      console.error("pitcher index failed:", err);
      await askCard({
        layout: "stack", tone: "error", title: "Couldn't reach the pitch data", text: "Check your connection and try again.",
        buttons: [{ label: "Try again", big: true, value: "retry" }],
      });
    }
  }
}

/** B: a link names a pitcher who is not in the index. Resolves once the visitor chooses to pick one. */
export const askNotInPark = () => askCard({
  layout: "stack", title: "That pitcher isn't in the park", text: "The link may be old. Here's who's pitching today:",
  buttons: [{ label: "Pick a pitcher", big: true, value: "pick" }],
});

/**
 * C: the GPU dropped the context. three.js handles the lost/restored events itself and re-uploads
 * geometry, textures, programs and render targets (the edge pass's float target included) on the next
 * render; this adds the visible side: a card while it is gone, `onRestored` to redraw, and a Reload
 * that becomes the main button if the context has not come back after 4 s.
 */
export function watchContext(canvas, { onLost, onRestored }) {
  let card = null, timer = 0;
  canvas.addEventListener("webglcontextlost", (e) => {
    e.preventDefault(); // without this the browser never restores the context
    onLost?.();
    if (card) return;
    card = showCard({
      spinner: true, title: "The picture paused", text: "Your graphics card reset. Restoring…",
      buttons: [{ label: "Reload", keep: true, onClick: () => location.reload() }],
    });
    timer = setTimeout(() => {
      card?.card.querySelector(".msg-text span").replaceChildren("Still not back. Reloading the page should fix it.");
      card?.buttons[0].classList.add("big");
      card?.buttons[0].focus();
    }, 4000);
  });
  canvas.addEventListener("webglcontextrestored", () => {
    clearTimeout(timer);
    card?.close();
    card = null;
    onRestored?.();
  });
}

/** Which pitchers have a comparison: one HEAD per index entry (a SPA fallback answers 200 with HTML, so check the type). */
async function comparisonSlugs(index) {
  const ok = await Promise.all(index.map(async (p) => {
    try {
      const r = await fetch(`/data/deliveries/${p.slug}.compare.json`, { method: "HEAD" });
      return r.ok && !!r.headers.get("content-type")?.includes("json") && p.slug;
    } catch { return false; }
  }));
  return new Set(ok.filter(Boolean));
}

/** D: the compare page has nothing for `slug`: say so kindly and list who does have one. */
export async function showNoCompare(slug) {
  console.warn(`no delivery / comparison data for ${slug}: run scripts/m5_build.py and scripts/m5_compare2d.py`);
  let index = [], list = [];
  try { index = await loadIndex(); list = compareChoices(index, await comparisonSlugs(index), slug); } catch (err) { console.warn("pitcher list not loaded:", err); }
  const name = displayName(index.find((p) => p.slug === slug)?.name ?? slug);
  showCard({
    layout: "stack", modal: false, focus: true,
    title: `No video comparison for ${name} yet`,
    text: list.length ? "His delivery is still being rebuilt from broadcast clips. These pitchers have one:" : "His delivery is still being rebuilt from broadcast clips.",
    choices: list.map((p) => ({ label: p.name, onClick: () => { location.href = `/compare.html?slug=${p.slug}`; } })),
    buttons: [{ label: "Back to the park", big: true, onClick: () => { location.href = "/"; } }],
  });
}

/** F: offer lighter graphics (non-modal: the pitch keeps playing; no focus grab, so Space still throws). */
export const offerLite = (onUse) => showCard({
  pos: "corner", modal: false, dismissible: true,
  title: "Running a bit slow?", text: "Lighter graphics keep the pitches smooth on this device.",
  buttons: [{ label: "Use lighter graphics", big: true, onClick: onUse }, { label: "Keep" }],
});

/** H: a one-time note on phones (remembered; shown once even if never dismissed). */
export function phoneNote(storage) {
  const coarse = typeof matchMedia === "function" && matchMedia("(pointer: coarse)").matches;
  if (!isPhone({ coarse, width: innerWidth, height: innerHeight }) || getFlag(PHONE_KEY, storage)) return;
  setFlag(PHONE_KEY, true, storage);
  showCard({
    pos: "bottom", modal: false, dismissible: true,
    title: "Best on a bigger screen", text: "Everything works here, but the camera views shine on a laptop or desktop.",
    buttons: [{ label: "Got it", big: true }],
  });
}
