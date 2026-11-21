// Start-screen data shaping (pure; the DOM lives in main.js).

/** Show the menu only when the URL does not already name a pitcher (deep links, screenshot scripts). */
export const showMenu = (search) => !new URLSearchParams(search).has("pitcher");

/** 76 -> 6'4" ; "" when unknown. */
export const heightLabel = (inches) =>
  Number.isFinite(inches) ? `${Math.floor(inches / 12)}'${Math.round(inches % 12)}"` : "";

/** "Sale, Chris" -> "Chris Sale". */
export const displayName = (name) => name.split(", ").reverse().join(" ");

const MIX_SHOWN = 5;
/**
 * Pitch mix from every pitch in the export, as whole percents that sum to exactly 100 (largest
 * remainder), biggest first; types past the top MIX_SHOWN fold into "Other".
 * @returns {{type: string, pct: number}[]}
 */
export function pitchMix(data) {
  const counts = {};
  for (const ab of data.at_bats) for (const p of ab.pitches) if (p.pitch_type) counts[p.pitch_type] = (counts[p.pitch_type] ?? 0) + 1;
  return mixFromCounts(counts);
}

/** Same as pitchMix from index.json's per-type pitch counts ({FF: 120, ...}); empty/missing gives none. */
export function mixFromCounts(counts = {}) {
  const total = Object.values(counts).reduce((s, n) => s + n, 0);
  if (!total) return [];
  let rows = Object.entries(counts).sort((a, b) => b[1] - a[1]);
  const rest = rows.slice(MIX_SHOWN).reduce((s, r) => s + r[1], 0);
  rows = rows.slice(0, MIX_SHOWN);
  if (rest) rows.push(["Other", rest]);
  const out = rows.map(([type, n]) => ({ type, pct: Math.floor((100 * n) / total), frac: ((100 * n) / total) % 1 }));
  let left = 100 - out.reduce((s, r) => s + r.pct, 0);
  for (const r of [...out].sort((a, b) => b.frac - a.frac)) if (left-- > 0) r.pct++;
  return out.map(({ type, pct }) => ({ type, pct }));
}

/** One card per pitcher in the index (a repeated slug keeps its last row). */
export function cardModels(index, deliverySlugs) {
  const bySlug = new Map(index.map((p) => [p.slug, p]));
  return [...bySlug.values()].map((p) => ({
    slug: p.slug, name: displayName(p.name), hand: `${p.throws}HP`, nAtBats: p.n_at_bats,
    height: heightLabel(p.height_in), mix: mixFromCounts(p.mix),
    fullDelivery: deliverySlugs.has(p.slug),
  }));
}

/**
 * Deep link into Sim: ?face=1&stand=L|R&batter=none|avg&call=0|1 -> the options it names
 * (the rest left to the defaults), or null when ?face=1 is absent. Unknown values are ignored.
 */
export function faceFromUrl(search) {
  const q = new URLSearchParams(search);
  if (q.get("face") !== "1") return null;
  const opts = {};
  if (["L", "R"].includes(q.get("stand"))) opts.stand = q.get("stand");
  if (["none", "avg"].includes(q.get("batter"))) opts.batter = q.get("batter");
  if (["0", "1"].includes(q.get("call"))) opts.call = q.get("call") === "1";
  return opts;
}
