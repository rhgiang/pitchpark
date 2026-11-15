// Exports from scripts/export_viewer.py, served by Vite from public/data/.

/** List of {slug, name, throws, n_at_bats}. */
export async function loadIndex() {
  return getJson("/data/index.json");
}

/** One pitcher's export (meta, pitcher, pitch_types, at_bats). */
export async function loadPitcher(slug) {
  return getJson(`/data/${slug}.json`);
}

/** One pitcher's delivery (M5), or null if there is none (a 404 is expected, not an error). */
export async function loadDelivery(slug) {
  const res = await fetch(`/data/deliveries/${slug}.json`);
  // Vite's SPA fallback can answer a missing file with 200 + index.html, so check the type too.
  if (!res.ok || !res.headers.get("content-type")?.includes("json")) return null;
  return res.json();
}

async function getJson(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  return res.json();
}
