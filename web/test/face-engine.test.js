import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { rng, typeProbs, locationMean, sampleTarget, hitterProbs, isStrike, zoneBin, reaim,
  newAtBat, throwPitch, nextCount } from "../src/face/engine.js";

// SYNTHETIC fixture (synthetic_at_bats): engine tests only, never real results.
const face = JSON.parse(readFileSync(new URL("./fixtures/face-synthetic.json", import.meta.url)));
const ctx0 = { count: "0-0", stand: "R", prevType: "start", tto: "1" };

test("parity with Python: type probs, location mean, zone bin, hitter probs, reaim", () => {
  let eP = 0, eL = 0, eH = 0, eR = 0;
  for (const c of face.parity) {
    const p = typeProbs(face, c.ctx);
    for (const [t, v] of Object.entries(c.type_probs)) eP = Math.max(eP, Math.abs(p[t] - v));
    const m = locationMean(face, c.type, c.ctx);
    eL = Math.max(eL, Math.hypot(m[0] - c.loc_mean[0], m[1] - c.loc_mean[1]));
    const h = c.hitter, zn = face.zone[c.ctx.stand];
    assert.equal(zn.top_ft, h.sz_top);
    assert.equal(zoneBin(h.x, h.z, h.sz_top, h.sz_bot), h.zone_bin);
    const hp = hitterProbs(face, c.ctx.stand, c.ctx, c.type, [h.x, h.z]);
    ["take", "whiff", "foul", "in_play"].forEach((k, i) => { eH = Math.max(eH, Math.abs(hp[k] - h.probs[i])); });
    const sim = reaim(face.bank[c.reaim.type][c.reaim.index].sim, c.reaim.dx, c.reaim.dz);
    assert.equal(sim.s.length, c.reaim.knots.s.length);
    sim.s.forEach((row, i) => row.forEach((v, j) => { eR = Math.max(eR, Math.abs(v - c.reaim.knots.s[i][j])); }));
    assert.equal(sim.T, c.reaim.knots.T);
  }
  console.log(`parity max err: probs ${eP} loc ${eL} hitter ${eH} reaim ${eR}`);
  assert.ok(eP < 1e-9 && eL < 1e-9 && eH < 1e-9 && eR < 1e-6);
});

test("parity covers unseen levels and on-edge zone values", () => {
  const prevs = face.parity.map((c) => c.ctx.prevType);
  assert.ok(prevs.includes("ZZ"));
  assert.ok(face.parity.some((c) => c.ctx.tto === "9") && face.parity.some((c) => c.ctx.count === "9-9"));
  assert.ok(face.parity.some((c) => c.hitter.zone_bin === "edge_in" && Math.abs(c.hitter.x) > 0.7));
});

test("zoneBin: on-edge goes to the lower bin (np.digitize right=True)", () => {
  const hw = 17 / 24 + 1.45 / 12, top = 3.5, bot = 1.6;
  assert.equal(zoneBin(hw, 2.5, top, bot), "edge_in"); // d == 0 exactly
  assert.equal(zoneBin(0, 2.5, top, bot), "heart");
  assert.equal(zoneBin(hw + 2 / 12, 2.5, top, bot), "edge_out");
  assert.equal(zoneBin(hw + 5 / 12, 2.5, top, bot), "chase");
  assert.equal(zoneBin(hw + 1, 2.5, top, bot), "waste");
});

test("isStrike: plate half-width plus a ball radius, zone grown the same way", () => {
  const zn = face.zone.R, hw = 17 / 24 + 1.45 / 12;
  assert.equal(isStrike(face, "R", [0, 2.5]), true);
  assert.equal(isStrike(face, "R", [hw, 2.5]), true);
  assert.equal(isStrike(face, "R", [hw + 0.01, 2.5]), false);
  assert.equal(isStrike(face, "R", [0, zn.top_ft + 1.45 / 12 + 0.01]), false);
  assert.equal(isStrike(face, "R", [0, zn.bot_ft - 1.45 / 12 - 0.01]), false);
});

test("typeProbs renormalises over bank types", () => {
  const p = typeProbs(face, ctx0);
  assert.deepEqual(Object.keys(p).sort(), Object.keys(face.bank).sort());
  assert.ok(Math.abs(Object.values(p).reduce((a, b) => a + b) - 1) < 1e-12);
});

test("sampled types match typeProbs (chi-square, 20k draws, seed 1)", () => {
  const p = typeProbs(face, ctx0), r = rng(1), n = 20000, cnt = {};
  for (let i = 0; i < n; i++) {
    const t = throwPitch(face, newAtBat(face, { stand: "R", batter: "none" }), r).pitch.pitch_type;
    cnt[t] = (cnt[t] || 0) + 1;
  }
  let chi = 0;
  for (const t of Object.keys(p)) chi += (((cnt[t] || 0) - n * p[t]) ** 2) / (n * p[t]);
  // df = types - 1 = 3; the chi-square critical value at p = 0.001 is 16.27
  assert.ok(chi < 16.27, `chi2 ${chi}`);
});

test("sampled targets match mean and cov (20k draws, 3 se)", () => {
  const t = "FF", n = 20000, r = rng(2), m = locationMean(face, t, ctx0), c = face.location[t].cov;
  const xs = Array.from({ length: n }, () => sampleTarget(face, t, ctx0, r));
  const mean = [0, 1].map((k) => xs.reduce((a, v) => a + v[k], 0) / n);
  for (const k of [0, 1]) assert.ok(Math.abs(mean[k] - m[k]) < 3 * Math.sqrt(c[k][k] / n));
  const cov = (a, b) => xs.reduce((s, v) => s + (v[a] - mean[a]) * (v[b] - mean[b]), 0) / (n - 1);
  for (const [a, b] of [[0, 0], [1, 1], [0, 1]]) {
    assert.ok(Math.abs(cov(a, b) - c[a][b]) < 0.05 * Math.sqrt(c[0][0] * c[1][1]), `cov ${a}${b}`);
  }
});

test("two-strike foul keeps count, foul tip strikes out", () => {
  assert.deepEqual(nextCount({ balls: 1, strikes: 2 }, "foul", true), { balls: 1, strikes: 2, event: null });
  assert.equal(nextCount({ balls: 1, strikes: 2 }, "whiff", true).event, "strikeout");
  assert.deepEqual(nextCount({ balls: 1, strikes: 1 }, "foul", true), { balls: 1, strikes: 2, event: null });
});

test("ball four walks, called third strike strikes out, in play ends", () => {
  assert.equal(nextCount({ balls: 3, strikes: 0 }, "take", false).event, "walk");
  assert.deepEqual(nextCount({ balls: 2, strikes: 1 }, "take", false), { balls: 3, strikes: 1, event: null });
  assert.equal(nextCount({ balls: 0, strikes: 2 }, "take", true).event, "strikeout");
  assert.equal(nextCount({ balls: 0, strikes: 0 }, "in_play", false).event, "in_play");
});

test("batter none: every result is a take", () => {
  const r = rng(3);
  for (let i = 0; i < 300; i++) {
    let ab = newAtBat(face, { stand: "L", batter: "none" });
    while (!ab.event) {
      const o = throwPitch(face, ab, r);
      ab = o.ab;
      assert.equal(o.pitch.outcome, "take");
      assert.ok(["ball", "called_strike"].includes(o.pitch.description));
    }
    assert.ok(["walk", "strikeout"].includes(ab.event));
  }
});

test("avg batter: at-bats end and every kind of result occurs", () => {
  const r = rng(4), seen = new Set();
  for (let i = 0; i < 300; i++) {
    let ab = newAtBat(face, { stand: "R", batter: "avg" });
    for (let n = 0; !ab.event; n++) {
      assert.ok(n < 40);
      const o = throwPitch(face, ab, r);
      ab = o.ab;
      seen.add(o.pitch.description);
    }
  }
  for (const d of ["ball", "called_strike", "swinging_strike", "foul", "hit_into_play"]) assert.ok(seen.has(d), d);
});

test("far target re-aims to the exact plate point", () => {
  const e = face.bank.FF[0], out = reaim(e.sim, 3, 0), a = e.sim.s.at(-1), b = out.s.at(-1);
  assert.ok(out.s.flat().every(Number.isFinite));
  assert.ok(Math.abs(b[0] - a[0] - 3) < 1e-9 && Math.abs(b[2] - a[2]) < 1e-9);
  assert.deepEqual(out.s[0].slice(0, 3), e.sim.s[0].slice(0, 3)); // release untouched
});

test("same seed, same at-bat", () => {
  const run = () => {
    const r = rng(9);
    let ab = newAtBat(face, { stand: "R", batter: "avg" });
    while (!ab.event) ab = throwPitch(face, ab, r).ab;
    return JSON.stringify(ab);
  };
  assert.equal(run(), run());
});

test("generated pitch is labelled, has viewer fields and no real path", () => {
  const { ab, pitch } = throwPitch(face, newAtBat(face, { stand: "R", batter: "none" }), rng(1));
  assert.equal(pitch.generated, true);
  assert.equal(pitch.real, null);
  assert.equal(pitch.status, "ok");
  assert.equal(ab.generated, true);
  assert.equal(ab.pitches.length, 1);
  for (const k of ["pitch_type", "pitch_name", "speed_mph", "spin_rpm", "spin_axis_deg", "arm_angle_deg", "extension_ft",
    "release", "sz_top_ft", "sz_bot_ft", "sim", "plate_sim", "balls", "strikes", "description", "outcome", "why"]) {
    assert.ok(k in pitch, k);
  }
  assert.ok("break_x_in" in pitch.tunnel && "break_z_in" in pitch.tunnel);
  assert.ok(Math.abs(pitch.sim.s.at(-1)[0] - pitch.plate_sim[0]) < 1e-3);
  assert.deepEqual(Object.keys(pitch.why).sort(), ["count", "p", "stand"]);
});
