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
const RIGHT_LEG = /^mixamorig:Right(Leg|Foot|ToeBase)/; // the airborne drive leg
const WINDOWS = [[-0.13, 0.03], [0.14, 0.46]]; // around release, and mid follow-through (measured)
const inWindow = (t) => WINDOWS.some(([a, b]) => t >= a && t <= b);
after(() => { frozen.legfix.on = true; });

test("legfix ON vs OFF on Ohtani: only the right leg moves, only inside the event windows", (t) => {
  const d = load("../public/data/deliveries/ohtani-shohei.json");
  if (!d) return t.skip("web/public/data not present");
  const fig = frozen.makePitcher(d.throws, d.height_in);
  let outside = 0, inside = 0, otherBones = 0;
  for (const release of [[-2.4, 54.3, 5.7], [2.9, 54.2, 5.0]]) {
    for (let s = -d.pre_s; s <= d.post_s; s += 1 / 120) {
      frozen.legfix.on = false; frozen.poseAtTime(fig, d, s, release, 35); const off = pos(fig);
      frozen.legfix.on = true; frozen.poseAtTime(fig, d, s, release, 35); const on = pos(fig);
      for (const k in off) {
        const dev = off[k].distanceTo(on[k]);
        if (!RIGHT_LEG.test(k)) otherBones = Math.max(otherBones, dev);
        else if (inWindow(s)) inside = Math.max(inside, dev);
        else outside = Math.max(outside, dev);
      }
    }
  }
  console.log(`legfix deviation (ft): other bones ${otherBones}, right leg outside windows ${outside}, inside ${inside.toFixed(4)}`);
  assert.ok(otherBones < 1e-6 && outside < 1e-6, `${otherBones} ${outside}`);
  assert.ok(inside > 0.01 && inside < 0.3, `inside ${inside}`);
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
