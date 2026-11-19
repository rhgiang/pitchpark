import { expandPitcher } from "./trajectory.js";

// Exports from scripts/export_viewer.py, served by Vite from public/data/.

/** List of {slug, name, throws, n_at_bats, height_in, mix: {pitch_type: count}}. */
export async function loadIndex() {
  return getJson("/data/index.json");
}

// One in-flight/finished request per file, so a hover prefetch and the real load are the same download.
// A failure is dropped from the cache so "Try again" really retries.
const cache = new Map();
function once(url, load) {
  if (!cache.has(url)) cache.set(url, load().catch((err) => { cache.delete(url); throw err; }));
  return cache.get(url);
}

/** One pitcher's export (meta, pitcher, pitch_types, at_bats). Rejects on a missing file or unknown schema. */
export function loadPitcher(slug) {
  return once(`/data/${slug}.json`, async () => expandPitcher(await getJson(`/data/${slug}.json`)));
}

/** Start fetching a pitcher's data and delivery ahead of the pick (hover, focus, selection); never throws. */
export function prefetchPitcher(slug) {
  loadPitcher(slug).catch(() => {}); // the real load reports the error
  loadDelivery(slug);
}

/** One pitcher's delivery (M5), or null if there is none (a 404 is expected, not an error). */
export const loadDelivery = (slug) => once(`/data/deliveries/${slug}.json`, () => fetchDelivery(slug));
async function fetchDelivery(slug) {
  // The delivery is optional: any failure (network, malformed file) falls back to the static pose
  // instead of failing the whole pitcher load.
  try {
    const res = await fetch(`/data/deliveries/${slug}.json`);
    // Vite's SPA fallback can answer a missing file with 200 + index.html, so check the type too.
    if (!res.ok || !res.headers.get("content-type")?.includes("json")) return null;
    const d = await res.json();
    return Number.isFinite(d?.pre_s) ? d : null;
  } catch (err) {
    console.warn(`delivery for ${slug} not loaded:`, err);
    return null;
  }
}

async function getJson(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  return res.json();
}
