// Exports from scripts/export_viewer.py, served by Vite from public/data/.

/** List of {slug, name, throws, n_at_bats}. */
export async function loadIndex() {
  return getJson("/data/index.json");
}

/** One pitcher's export (meta, pitcher, pitch_types, at_bats). */
export async function loadPitcher(slug) {
  return getJson(`/data/${slug}.json`);
}

async function getJson(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  return res.json();
}
