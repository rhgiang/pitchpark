// Ohtani leg-jitter fix (frozen path, legfix switch). OFF must be exactly 70c9f22 (signed_off.test.js
// checks that with legfix.on = false); ON may only move the airborne
// drive leg, only where the delivery's foot is beyond the rig's reach.
import { test, before, after } from "node:test";
import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import { Vector3 } from "three";
import * as frozen from "../src/signed_off/pitcher.js";
import { J } from "../src/signed_off/skeleton.js";

const load = (rel) => { const u = new URL(rel, import.meta.url); return existsSync(u) ? JSON.parse(readFileSync(u)) : null; };
const pos = (fig) => {
  fig.updateMatrixWorld(true);
  const o = {};
  fig.traverse((b) => { if (b.name.startsWith("mixamorig:")) o[b.name] = b.getWorldPosition(new Vector3()); });
  return o;
};
const kneeDeg = (fig, sd) => {
  const g = (n) => fig.userData.bones[J[`${sd}_${n}`]].getWorldPosition(new Vector3());
  return g("hip").sub(g("knee")).angleTo(g("ankle").sub(g("knee"))) * 180 / Math.PI;
};
// Below-the-hip bones: the only ones legfix may move (hips, spine, arms and the release hand must not).
const SHIN_FOOT = /^mixamorig:(Left|Right)(Leg|Foot|ToeBase)/;
const WINDOWS = [[-1.3, -0.7], [-0.3, 0.6]]; // leg lift peak, and foot strike -> follow-through (measured)
const inWindow = (t) => WINDOWS.some(([a, b]) => t >= a && t <= b);
after(() => { frozen.legfix.on = true; });

test("legfix ON vs OFF on Ohtani: only shin/foot/toe bones move; hips, spine, arms, release hand do not", (t) => {
  const d = load("../public/data/deliveries/ohtani-shohei.json");
  if (!d) return t.skip("web/public/data not present");
  const fig = frozen.makePitcher(d.throws, d.height_in);
  let outside = 0, inside = 0, otherBones = 0, hand = 0, plantedToe = 0;
  for (const release of [[-2.4, 54.3, 5.7], [2.9, 54.2, 5.0]]) {
    for (let s = -d.pre_s; s <= d.post_s; s += 1 / 120) {
      frozen.legfix.on = false; frozen.poseAtTime(fig, d, s, release, 35); const off = pos(fig), hOff = frozen.handPosition(fig).clone();
      frozen.legfix.on = true; frozen.poseAtTime(fig, d, s, release, 35); const on = pos(fig), hOn = frozen.handPosition(fig).clone();
      hand = Math.max(hand, hOff.distanceTo(hOn));
      for (const k in off) {
        const dev = off[k].distanceTo(on[k]);
        if (!SHIN_FOOT.test(k)) otherBones = Math.max(otherBones, dev);
        else if (inWindow(s)) inside = Math.max(inside, dev);
        else outside = Math.max(outside, dev);
        // the planted lead foot (left) stays where it was: ankle and toe, from foot strike on
        if (/LeftFoot|LeftToeBase/.test(k) && s >= -0.1) plantedToe = Math.max(plantedToe, dev);
      }
    }
  }
  console.log(`legfix deviation (ft): non-shin bones ${otherBones}, release hand ${hand}, shin/foot outside windows ${outside.toFixed(4)}, inside ${inside.toFixed(3)}, planted lead foot ${plantedToe.toFixed(4)}`);
  assert.ok(otherBones < 1e-6 && hand < 1e-6, `other ${otherBones} hand ${hand}`);
  assert.ok(outside < 0.05, `outside ${outside}`);
  assert.ok(plantedToe < 0.02, `planted foot moved ${plantedToe}`);
  assert.ok(inside > 0.01 && inside < 1.3, `inside ${inside}`);
});

test("legfix: knee no longer snaps when an airborne foot runs out of reach (SYNTHETIC delivery)", () => {
  // SYNTHETIC: the standing test fixture with the right ankle/foot swung out sideways and up
  // (smoothstep in, hold, out), so the delivery's foot is beyond the rig's leg for a while.
  const d = structuredClone(load("./fixtures/delivery-synthetic.json"));
  const ss = (x) => (x <= 0 ? 0 : x >= 1 ? 1 : x * x * (3 - 2 * x));
  d.joints_ft.forEach((fr, f) => {
    const a = ss((f - 25) / 15) * (1 - ss((f - 55) / 15));
    for (const j of [8, 11]) { fr[j][0] += 2.4 * a; fr[j][2] += 0.9 * a; }
  });
  const fig = frozen.makePitcher(d.throws, d.height_in);
  const worst = (on) => {
    frozen.legfix.on = on;
    let w = 0, prev = [];
    for (let s = -0.6; s <= 0.4; s += 1 / 240) {
      frozen.poseAtTime(fig, d, s, null, 35);
      prev.push(kneeDeg(fig, "right"));
      const n = prev.length; if (n >= 3) w = Math.max(w, Math.abs(prev[n - 1] - 2 * prev[n - 2] + prev[n - 3]));
    }
    return w;
  };
  const off = worst(false), on = worst(true);
  console.log(`synthetic knee max 2nd difference (deg/frame^2): off ${off.toFixed(2)}, on ${on.toFixed(2)}`);
  assert.ok(off > 3 * on && on < 1, `off ${off} on ${on}`);
});

test("legfix ON: Ohtani's drive-knee 2nd difference stays small at the four lock/unlock events", (t) => {
  const d = load("../public/data/deliveries/ohtani-shohei.json");
  if (!d) return t.skip("web/public/data not present");
  const fig = frozen.makePitcher(d.throws, d.height_in);
  const series = (on) => {
    frozen.legfix.on = on;
    const out = [];
    for (let s = -0.15; s <= 0.45; s += 1 / 240) { frozen.poseAtTime(fig, d, s, [-2.4, 54.3, 5.7], 40); out.push([s, kneeDeg(fig, "right")]); }
    return out;
  };
  const maxD2 = (a) => Math.max(...a.slice(2).map((_, i) => Math.abs(a[i + 2][1] - 2 * a[i + 1][1] + a[i][1])));
  const off = maxD2(series(false)), on = maxD2(series(true));
  console.log(`Ohtani right knee max 2nd difference (deg/frame^2, -0.15..0.45 s): off ${off.toFixed(2)}, on ${on.toFixed(2)}`);
  assert.ok(off > 10 && on < 3.5, `off ${off} on ${on}`);
});

test("legfix: Ohtani leg stutter metrics before/after (delivery-driven stutters, 1/240 s)", (t) => {
  const d = load("../public/data/deliveries/ohtani-shohei.json");
  if (!d) return t.skip("web/public/data not present");
  const fig = frozen.makePitcher(d.throws, d.height_in);
  const run = (on) => {
    frozen.legfix.on = on;
    const g = (n) => fig.userData.bones[J[n]].getWorldPosition(new Vector3());
    const S = { lk: [], rk: [], ra: [], ang: [] };
    for (let s = -0.3; s <= 0.5; s += 1 / 240) {
      frozen.poseAtTime(fig, d, s, [-2.4, 54.3, 5.7], 35); fig.updateMatrixWorld(true);
      S.lk.push(g("left_knee")); S.ra.push(g("right_ankle")); S.ang.push(kneeDeg(fig, "right"));
    }
    const dt = 1 / 240;
    const vel = (a) => a.slice(1).map((p, i) => p.clone().sub(a[i]).divideScalar(dt));
    const acc = (a) => vel(a).slice(1).map((v, i, V) => v.clone().sub(vel(a)[i]).divideScalar(dt));
    const mx = (a) => Math.max(...a.map((v) => v.length()));
    const d2 = S.ang.slice(2).map((_, i) => Math.abs(S.ang[i + 2] - 2 * S.ang[i + 1] + S.ang[i]));
    return { leadKneeSpeed: mx(vel(S.lk)), driveAnkleAccel: mx(acc(S.ra)), driveKneeD2: Math.max(...d2) };
  };
  const off = run(false), on = run(true);
  console.log("stutter metrics -0.3..0.5 s, off -> on:", Object.keys(off).map((k) => `${k} ${off[k].toFixed(2)} -> ${on[k].toFixed(2)}`).join("; "));
  assert.ok(on.leadKneeSpeed < 0.75 * off.leadKneeSpeed, "lead knee swing at foot strike");     // 38 -> 26 ft/s (delivery 17)
  assert.ok(on.driveAnkleAccel < 0.6 * off.driveAnkleAccel, "drive foot toe-off kick");         // 1504 -> ~660 ft/s^2
  assert.ok(on.driveKneeD2 < 0.25 * off.driveKneeD2 && on.driveKneeD2 < 1.5, "drive knee lock/snap"); // 6.45 -> ~0.84
});
