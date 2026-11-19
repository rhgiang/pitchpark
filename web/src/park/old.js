// ?old=crowd,home,structure (comma list): put the park's older look back, part by part, for A/B.
// Evaluated when asked (not at import), so tests and a URL change both work.
export const isOld = (name, search = globalThis.location?.search ?? "") =>
  (new URLSearchParams(search).get("old") ?? "").split(",").includes(name);
