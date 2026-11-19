// Old (schema 2, 5 ms samples) vs new (schema 3, fit + Hermite knots) pitch exports.
//   node tools/compare-exports.mjs <old dir> <new dir>
// For every pitch of every pitcher: max position error (ft) of the new expanded path against the old
// samples at the old sample times, per path kind, and plate-time / plate-position differences.
import { readFileSync, readdirSync } from "node:fs";
import { expandPitcher } from "../src/trajectory.js";

const [oldDir, newDir] = process.argv.slice(2);
const slugs = readdirSync(oldDir).filter((f) => f.endsWith(".json") && f !== "index.json");
let worst = { real: 0, sim: 0, nospin: 0, plate: 0, time: 0 };
console.log("slug                 pitches   real_ft   sim_ft   nospin_ft  plate_ft  plate_s");
for (const f of slugs) {
  const o = JSON.parse(readFileSync(`${oldDir}/${f}`, "utf8"));
  const n = expandPitcher(JSON.parse(readFileSync(`${newDir}/${f}`, "utf8")));
  const op = o.at_bats.flatMap((a) => a.pitches), np = n.at_bats.flatMap((a) => a.pitches);
  if (op.length !== np.length) throw new Error(`${f}: ${op.length} vs ${np.length} pitches`);
  const m = { real: 0, sim: 0, nospin: 0, plate: 0, time: 0 };
  op.forEach((a, i) => {
    const b = np[i];
    if (a.pitch_number !== b.pitch_number || a.status !== b.status) throw new Error(`${f} pitch ${i} differs`);
    for (const k of ["real", "sim", "nospin"]) {
      if (!a[k]) { if (b[k]) throw new Error(`${f} ${i} ${k} appeared`); continue; }
      const q = b[k];
      // every old sample time, evaluated by the new path: linear between new 5 ms samples (like the viewer)
      let j = 0;
      // The last sample is compared point to point: the old file rounded its time to 1e-4 s (up to
      // 5e-5 s x ~130 ft/s = 0.0065 ft off the position it holds), the new one keeps 1e-6 s.
      const last = a[k].t.length - 1;
      for (let c = 0; c < 3; c++) m[k] = Math.max(m[k], Math.abs(q.p.at(-1)[c] - a[k].p[last][c]));
      a[k].t.slice(0, last).forEach((t, s) => {
        while (j < q.t.length - 2 && q.t[j + 1] <= t) j++;
        const w = Math.min(1, Math.max(0, (t - q.t[j]) / (q.t[j + 1] - q.t[j])));
        for (let c = 0; c < 3; c++) m[k] = Math.max(m[k], Math.abs(q.p[j][c] + w * (q.p[j + 1][c] - q.p[j][c]) - a[k].p[s][c]));
      });
      m.time = Math.max(m.time, Math.abs(q.t.at(-1) - a[k].t.at(-1)));
    }
    for (const k of ["plate_real", "plate_sim", "plate_nospin"]) {
      if (!a[k]) continue;
      m.plate = Math.max(m.plate, Math.abs(a[k][0] - b[k][0]), Math.abs(a[k][1] - b[k][1]));
      m.time = Math.max(m.time, Math.abs(a[k][2] - b[k][2]));
    }
  });
  for (const k in m) worst[k] = Math.max(worst[k], m[k]);
  console.log(`${f.replace(".json", "").padEnd(20)} ${String(op.length).padStart(7)}  ${[m.real, m.sim, m.nospin, m.plate].map((x) => x.toFixed(5).padStart(9)).join(" ")} ${m.time.toExponential(1).padStart(9)}`);
}
console.log(`MAX over all: real ${worst.real.toFixed(5)} ft, sim ${worst.sim.toFixed(5)} ft, nospin ${worst.nospin.toFixed(5)} ft, plate position ${worst.plate.toFixed(5)} ft, time ${worst.time.toExponential(1)} s`);
if (Math.max(worst.real, worst.sim, worst.nospin, worst.plate) >= 0.01) { console.log("FAIL: over 0.01 ft"); process.exit(1); }
