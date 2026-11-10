// Spec check 1: the viewer shows what Python computed. Plays every exported Statcast-fit path
// through the viewer's own interpolation (stateAt) and compares with the exporter's numbers.
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync, readdirSync, existsSync } from "node:fs";
import { stateAt, duration } from "../src/trajectory.js";

const here = new URL(".", import.meta.url);
const TOL_FT = 0.01 / 12; // 0.01 in
const TOL_S = 1e-6;
const PLATE_Y_FT = 17 / 12; // front of the plate, where the exported paths end

function pitchesOf(file) {
  return JSON.parse(readFileSync(file, "utf8")).at_bats.flatMap((ab) => ab.pitches);
}

const files = [new URL("fixtures/synthetic.json", here)];
const dataDir = new URL("../public/data/", here);
if (existsSync(dataDir)) {
  for (const f of readdirSync(dataDir)) {
    if (f.endsWith(".json") && f !== "index.json") files.push(new URL(f, dataDir));
  }
}

test("Statcast-fit paths replay the exporter's analytic points and plate time", () => {
  let checked = 0;
  let worstIn = 0;
  for (const file of files) {
    for (const p of pitchesOf(file)) {
      if (!p.real) continue; // no_tracking: nothing to fly
      const T = duration(p.real);
      // Independent: check_real is evaluated from the 9-parameter fit, not from the samples.
      for (const [t, x, y, z] of p.check_real) {
        const s = stateAt(p.real, t).p;
        const d = Math.max(Math.abs(s[0] - x), Math.abs(s[1] - y), Math.abs(s[2] - z));
        worstIn = Math.max(worstIn, d * 12);
        assert.ok(d < TOL_FT, `${file.pathname} check_real t=${t}: off by ${(d * 12).toFixed(4)} in`);
      }
      // Not independent either: the exporter writes round(flight, 4) as both the path's last t
      // and plate_real[2]. Kept as a guard that duration() reads the right sample.
      assert.ok(Math.abs(T - p.plate_real[2]) < TOL_S, `plate time ${T} vs ${p.plate_real[2]}`);
      // Not independent: the exporter copies plate_real's x/z into the last sample. Kept as a
      // guard that stateAt clamps at the end and that the path really stops at the plate.
      const end = stateAt(p.real, T);
      assert.ok(end.done);
      const [px, pz] = p.plate_real;
      assert.ok(Math.abs(end.p[0] - px) < TOL_FT && Math.abs(end.p[2] - pz) < TOL_FT);
      assert.ok(Math.abs(end.p[1] - PLATE_Y_FT) < TOL_FT);
      checked++;
    }
  }
  console.log(`truth test: ${checked} real pitches from ${files.length} files, largest check_real difference ${worstIn.toFixed(5)} in`);
  assert.ok(checked > 0);
});

test("no-spin paths replay the exporter's RK4 points and end at the plate", () => {
  let checked = 0;
  let worstIn = 0;
  for (const file of files) {
    for (const p of pitchesOf(file)) {
      if (!p.nospin) {
        assert.equal(p.status === "ok", false, "a modelled pitch must have a no-spin path");
        continue;
      }
      // check_nospin comes from the 1 ms RK4 output, not from the 5 ms samples played here.
      for (const [t, x, y, z] of p.check_nospin) {
        const s = stateAt(p.nospin, t).p;
        const d = Math.max(Math.abs(s[0] - x), Math.abs(s[1] - y), Math.abs(s[2] - z));
        worstIn = Math.max(worstIn, d * 12);
        assert.ok(d < TOL_FT, `${file.pathname} check_nospin t=${t}: off by ${(d * 12).toFixed(4)} in`);
      }
      const T = duration(p.nospin);
      assert.ok(Math.abs(T - p.plate_nospin[2]) < TOL_S);
      const end = stateAt(p.nospin, T);
      assert.ok(end.done && Math.abs(end.p[1] - PLATE_Y_FT) < TOL_FT);
      assert.ok(Math.abs(end.p[0] - p.plate_nospin[0]) < TOL_FT && Math.abs(end.p[2] - p.plate_nospin[1]) < TOL_FT);
      checked++;
    }
  }
  console.log(`no-spin truth test: ${checked} pitches, largest check_nospin difference ${worstIn.toFixed(5)} in`);
  assert.ok(checked > 0);
});

test("stateAt interpolates linearly and clamps at both ends", () => {
  const path = { t: [0, 1], p: [[0, 10, 0], [2, 0, 4]] };
  assert.deepEqual(stateAt(path, 0.5), { p: [1, 5, 2], done: false });
  assert.deepEqual(stateAt(path, 3), { p: [2, 0, 4], done: true });
  assert.deepEqual(stateAt(path, -1), { p: [0, 10, 0], done: false });
});

test("flight times come from the data, not a fixed 0.4 s", () => {
  const ps = pitchesOf(new URL("fixtures/synthetic.json", here));
  const cu = ps.find((p) => p.pitch_type === "CU" && p.real);
  const ff = ps.find((p) => p.pitch_type === "FF" && p.real);
  assert.ok(duration(cu.real) > 0.45, `CU ${duration(cu.real)}`);
  assert.ok(duration(ff.real) < duration(cu.real));
});
