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
  const counts = new Map();
  for (const ab of data.at_bats) for (const p of ab.pitches) if (p.pitch_type) counts.set(p.pitch_type, (counts.get(p.pitch_type) ?? 0) + 1);
  const total = [...counts.values()].reduce((s, n) => s + n, 0);
  if (!total) return [];
  let rows = [...counts].sort((a, b) => b[1] - a[1]);
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
    fullDelivery: deliverySlugs.has(p.slug),
  }));
}
