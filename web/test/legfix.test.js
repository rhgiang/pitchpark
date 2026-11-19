// Ohtani leg/body smoothing fixes (frozen path, legfix switch). OFF must be exactly 70c9f22 (signed_off.test.js
// checks that with legfix.on = false); ON re-places the pelvis, re-solves the legs and arm, and smooths the motion.
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

// Realistic release points for this RHP (Statcast ft); the arm must put the hand on them at t = 0.
const RELEASES = [[-2.4, 54.3, 5.7], [-1.5, 54.8, 6.3]];

test("legfix ON vs OFF on Ohtani: the planted lead foot stays planted, the release hand is on the release", (t) => {
  const d = load("../public/data/deliveries/ohtani-shohei.json");
  if (!d) return t.skip("web/public/data not present");
  const fig = frozen.makePitcher(d.throws, d.height_in);
  const offToe = {}, onToe = {};
  let plantedToe = 0, missOn = 0, missOff = 0;
  for (const release of RELEASES) {
    for (let s = -d.pre_s; s <= d.post_s; s += 1 / 120) {
      frozen.legfix.on = false; frozen.poseAtTime(fig, d, s, release, 35); const off = pos(fig);
      frozen.legfix.on = true; frozen.poseAtTime(fig, d, s, release, 35); const on = pos(fig);
      for (const k in off) if (/LeftFoot|LeftToeBase/.test(k) && s >= -0.1) plantedToe = Math.max(plantedToe, off[k].distanceTo(on[k]));
      if (s >= 0 && s <= 0.4) { for (const [m, o] of [[offToe, off], [onToe, on]]) { const q = o["mixamorig:LeftToeBase"]; m.first ??= q.clone(); m.drift = Math.max(m.drift ?? 0, q.distanceTo(m.first)); } }
    }
    for (const [on, set] of [[false, (v) => (missOff = Math.max(missOff, v))], [true, (v) => (missOn = Math.max(missOn, v))]]) {
      frozen.legfix.on = on; frozen.poseAtTime(fig, d, 0, release, 35);
      set(frozen.handPosition(fig).distanceTo(new Vector3(release[0], release[2], -release[1])));
    }
  }
  console.log(`legfix: planted lead foot off-vs-on ${plantedToe.toFixed(4)} ft, lead toe drift 0..0.4 s off ${offToe.drift.toFixed(3)} on ${onToe.drift.toFixed(3)}, release-hand miss off ${missOff.toFixed(4)} on ${missOn.toFixed(4)} ft`);
  assert.ok(plantedToe < 0.05, `planted foot moved ${plantedToe}`);
  assert.ok(onToe.drift < offToe.drift + 0.02 && onToe.drift < 0.1, `planted toe slides ${onToe.drift} (off ${offToe.drift})`); // stays put once planted (0..0.4 s)
  assert.ok(missOn < 0.05, `release miss ${missOn}`);
});

test("legfix: pelvis follows the delivery's height (no dip/pop), the arm is smoother and does not park at release", (t) => {
  const d = load("../public/data/deliveries/ohtani-shohei.json");
  if (!d) return t.skip("web/public/data not present");
  const fig = frozen.makePitcher(d.throws, d.height_in);
  const release = RELEASES[0], dt = 1 / 240, k = d.height_in / 75;
  const run = (on) => {
    frozen.legfix.on = on;
    const g = (n) => fig.userData.bones[J[n]].getWorldPosition(new Vector3());
    const S = { ts: [], pelvis: [], rw: [], re: [], rk: [], lk: [] };
    for (let s = -0.6; s <= 0.6; s += dt) {
      frozen.poseAtTime(fig, d, s, release, 35); fig.updateMatrixWorld(true);
      S.ts.push(s); S.pelvis.push(g("pelvis")); S.rw.push(g("right_wrist")); S.re.push(g("right_elbow")); S.rk.push(g("right_knee")); S.lk.push(g("left_knee"));
    }
    // delivery pelvis height (spline through its frames, ft above the same origin) for the dip
    const dz = S.ts.map((s) => { const f = d.release_index + s * d.fps, i = Math.floor(f), u = f - i; return d.joints_ft[i][J.pelvis][2] * (1 - u) + d.joints_ft[i + 1][J.pelvis][2] * u; });
    const dip = Math.max(...S.pelvis.map((p, i) => dz[i] - p.y));
    const jerk = (a) => Math.max(...a.slice(3).map((_, i) => a[i + 3].clone().sub(a[i + 2].clone().multiplyScalar(3)).add(a[i + 1].clone().multiplyScalar(3)).sub(a[i]).length() / dt ** 3));
    const at = (t0) => S.ts.findIndex((s) => s >= t0 - 1e-9);
    const speed = (a, t0) => a[at(t0) + 1].distanceTo(a[at(t0)]) / dt;
    return { dip, pelvisSpeed025: speed(S.pelvis, 0.25), wristSpeed0: speed(S.rw, 0), armJerk: Math.max(jerk(S.re), jerk(S.rw)), kneeJerk: Math.max(jerk(S.rk), jerk(S.lk)) };
  };
  const off = run(false), on = run(true);
  console.log("pelvis/arm metrics off -> on:", Object.keys(off).map((n) => `${n} ${off[n].toFixed(2)} -> ${on[n].toFixed(2)}`).join("; "));
  assert.ok(off.dip > 1.2 && on.dip < 0.6, `pelvis dip below the delivery's: off ${off.dip} on ${on.dip}`);    // 1.6 -> 0.4 ft
  assert.ok(on.pelvisSpeed025 < 0.4 * off.pelvisSpeed025, "pelvis pop-up at +0.25 s");                           // 8.6 -> 1.6 ft/s
  assert.ok(on.wristSpeed0 > 15 && off.wristSpeed0 < 10, "the hand sweeps through release (was parked)");       // 2 -> 40 ft/s
  assert.ok(on.armJerk < 0.5 * off.armJerk, "arm jerk");                                                         // 1.3M -> 0.4M ft/s^3
  assert.ok(on.kneeJerk < off.kneeJerk, "knee jerk");                                                            // 3.8M -> 0.23M
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
