// M7 check, sim side: 20,000 Average-hitter at-bats per pitcher from the browser engine, the same
// statistics as scripts/m7_real_stats.py, compared to the real held-out interval.
//   node scripts/m7_check.mjs [slug|--all]  -> results/m7/<slug>-sim.json, results/m7/checks.csv
import { readFileSync, writeFileSync, readdirSync } from "node:fs";
import { rng, newAtBat, throwPitch, isStrike } from "../web/src/face/engine.js";

const ROOT = new URL("..", import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, "$1");
const OUT = `${ROOT}results/m7`, N_AB = 20000, SEED = 0;
const MIN_MIX_N = 20; // counts with fewer real pitches have a degenerate bootstrap TV threshold; skipped
const COUNTS = [...Array(4).keys()].flatMap((b) => [0, 1, 2].map((s) => `${b}-${s}`));
const FPS_OUT = new Set(["whiff", "foul", "in_play"]);
const arg = process.argv[2] ?? "--all";
const slugs = arg === "--all" ? readdirSync(`${OUT}`).filter((f) => f.endsWith("-real.json")).map((f) => f.replace("-real.json", "")) : [arg];

function simulate(face, shareL) {
  const r = rng(SEED), S = {};
  for (const s of ["L", "R"]) S[s] = { n_ab: 0, pitches: 0, zone: 0, fps: 0, end: { strikeout: 0, walk: 0, in_play: 0 }, mix: {} };
  for (let i = 0; i < N_AB; i++) {
    const stand = r() < shareL ? "L" : "R", st = S[stand];
    let ab = newAtBat(face, { stand, batter: "avg", count: "0-0", id: `sim-${i}` });
    while (!ab.event) {
      const { ab: nx, pitch } = throwPitch(face, ab, r);
      st.pitches++;
      if (isStrike(face, stand, pitch.plate_sim)) st.zone++;
      const c = `${pitch.balls}-${pitch.strikes}`;
      (st.mix[c] ??= {})[pitch.pitch_type] = (st.mix[c][pitch.pitch_type] ?? 0) + 1;
      // first-pitch strike: 0-0 pitch that is a strike call, swing-and-miss, foul or in play
      if (c === "0-0" && (FPS_OUT.has(pitch.outcome) || (pitch.outcome === "take" && isStrike(face, stand, pitch.plate_sim)))) st.fps++;
      ab = nx;
    }
    st.n_ab++; st.end[ab.event]++;
  }
  return S;
}

const rows = [["pitcher", "stand", "stat", "real", "lo", "hi", "sim", "inside"]];
for (const slug of slugs) {
  const real = JSON.parse(readFileSync(`${OUT}/${slug}-real.json`));
  const face = JSON.parse(readFileSync(`${ROOT}web/public/data/face/${slug}.json`));
  const t0 = Date.now(), S = simulate(face, real.stand_share.L.est), secs = (Date.now() - t0) / 1000;
  const sim = { slug, seconds: secs, n_ab: N_AB };
  const add = (stand, stat, e, v) => rows.push([slug, stand, stat, e.est, e.lo, e.hi, v, v >= e.lo && v <= e.hi ? 1 : 0]);
  for (const stand of ["L", "R"]) {
    const s = S[stand], rl = real[stand];
    if (!rl || !s.n_ab) continue;
    const m = { n_ab: s.n_ab, zone_rate: s.zone / s.pitches, first_pitch_strike: s.fps / s.n_ab, pitches_per_ab: s.pitches / s.n_ab,
      outcomes: Object.fromEntries(Object.entries(s.end).map(([k, v]) => [k, v / s.n_ab])), mix_by_count: {}, mix_tv: {} };
    add(stand, "zone_rate", rl.zone_rate, m.zone_rate);
    add(stand, "first_pitch_strike", rl.first_pitch_strike, m.first_pitch_strike);
    add(stand, "pitches_per_ab", rl.pitches_per_ab, m.pitches_per_ab);
    for (const o of Object.keys(m.outcomes)) add(stand, `outcome_${o}`, rl.outcomes[o], m.outcomes[o]);
    for (const c of COUNTS) {
      const mc = s.mix[c], rm = rl.mix_by_count[c];
      if (!mc || !rm || rl.mix_n[c] < MIN_MIX_N) continue;
      const n = Object.values(mc).reduce((a, b) => a + b, 0);
      m.mix_by_count[c] = Object.fromEntries(Object.keys(rm).map((t) => [t, (mc[t] ?? 0) / n]));
      // TV distance sim vs real point mix; threshold = real bootstrap's 95th percentile of TV
      const tv = 0.5 * Object.keys(rm).reduce((a, t) => a + Math.abs(m.mix_by_count[c][t] - rm[t]), 0);
      m.mix_tv[c] = tv;
      // mix_tv rows: real/hi = real-bootstrap TV95 threshold, lo = 0, sim = sim TV (no sim noise; fine at 20k ABs)
      add(stand, `mix_tv@${c}`, { est: rl.mix_tv95[c], lo: 0, hi: rl.mix_tv95[c] }, tv);
    }
    sim[stand] = m;
  }
  writeFileSync(`${OUT}/${slug}-sim.json`, JSON.stringify(sim, null, 1));
  console.log(`${slug}: ${N_AB} at-bats in ${secs.toFixed(1)} s`);
}
// --all rewrites checks.csv; a single slug appends into the existing file's other rows
let body = rows.slice(1).map((r) => r.join(","));
if (arg !== "--all") {
  try { body = [...readFileSync(`${OUT}/checks.csv`, "utf8").trim().split("\n").slice(1).filter((l) => !l.startsWith(arg + ",")), ...body]; } catch {}
}
writeFileSync(`${OUT}/checks.csv`, [rows[0].join(","), ...body].join("\n") + "\n");
