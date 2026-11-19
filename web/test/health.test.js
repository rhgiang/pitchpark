import { test } from "node:test";
import assert from "node:assert/strict";
import { getFlag, setFlag, LITE_KEY, canMakeWebGL, makeSlowDetector, makeAutoLite, shouldAutoLite, isPhone, slugInIndex, compareChoices, debugList } from "../src/health.js";
import { pixelRatio } from "../src/pacing.js";

const memory = () => { const m = new Map(); return { getItem: (k) => m.get(k) ?? null, setItem: (k, v) => m.set(k, v), removeItem: (k) => m.delete(k) }; };
const broken = { getItem() { throw new Error("blocked"); }, setItem() { throw new Error("blocked"); }, removeItem() { throw new Error("blocked"); } };

test("flags round-trip and never throw, even with blocked or missing storage", () => {
  const s = memory();
  assert.equal(getFlag(LITE_KEY, s), false);
  setFlag(LITE_KEY, true, s);
  assert.equal(getFlag(LITE_KEY, s), true);
  setFlag(LITE_KEY, false, s);
  assert.equal(getFlag(LITE_KEY, s), false);
  assert.equal(getFlag(LITE_KEY, broken), false);
  assert.doesNotThrow(() => { setFlag(LITE_KEY, true, broken); setFlag(LITE_KEY, false, broken); setFlag(LITE_KEY, true, null); });
  assert.equal(getFlag(LITE_KEY, null), false);
});

test("WebGL probe: a context means yes; null, a throw or no canvas means no", () => {
  assert.equal(canMakeWebGL(() => ({ getContext: () => ({}) })), true);
  assert.equal(canMakeWebGL(() => ({ getContext: () => null })), false);
  assert.equal(canMakeWebGL(() => ({ getContext: () => { throw new Error("no gl"); } })), false);
  assert.equal(canMakeWebGL(() => { throw new Error("no document"); }), false);
});

/** Feed frames every `ms` for `seconds` of active time; the first verdict (or false). */
function run(detector, ms, seconds, t0 = 0, active = true) {
  for (let t = t0; t <= t0 + seconds * 1000; t += ms) if (detector.frame(t, active)) return true;
  return false;
}

test("slow detector: ~20 fps for 5 s of active playback trips it; 60 and 30 fps do not", () => {
  assert.equal(run(makeSlowDetector(), 50, 6), true);
  assert.equal(run(makeSlowDetector(), 1000 / 60, 12), false);
  assert.equal(run(makeSlowDetector(), 1000 / 30, 12), false);
  assert.equal(run(makeSlowDetector(), 50, 4), false, "not enough active time yet");
});

test("slow detector: idle frames never count, active time adds up across pitches, stalls are skipped", () => {
  const d = makeSlowDetector();
  assert.equal(run(d, 50, 60, 0, false), false, "20 fps while idle is by design");
  // 2 s pitches at 20 fps with 10 s gaps between them: two (4 s active) are not enough, the third (6 s) is
  assert.equal(run(d, 50, 2, 0) || run(d, 50, 2, 12000), false);
  assert.equal(run(d, 50, 2, 24000), true);
  const s = makeSlowDetector();
  s.frame(0, true);
  s.frame(30000, true); // a 30 s stall (tab in the background): not a rate
  assert.equal(run(s, 1000 / 60, 12, 30000), false);
});

test("phones: touch-first and a small screen, landscape included", () => {
  assert.equal(isPhone({ coarse: true, width: 390, height: 844 }), true);
  assert.equal(isPhone({ coarse: true, width: 844, height: 390 }), true);
  assert.equal(isPhone({ coarse: true, width: 820, height: 1180 }), false, "tablet");
  assert.equal(isPhone({ coarse: false, width: 400, height: 800 }), false, "narrow desktop window");
});

test("slug lookup and the compare list (current pitcher left out, index order, display names)", () => {
  const index = [{ slug: "sale-chris", name: "Sale, Chris" }, { slug: "cease-dylan", name: "Cease, Dylan" }, { slug: "ohtani-shohei", name: "Ohtani, Shohei" }];
  assert.equal(slugInIndex(index, "sale-chris"), true);
  assert.equal(slugInIndex(index, "someone-else"), false);
  assert.deepEqual(compareChoices(index, new Set(["sale-chris", "ohtani-shohei", "ghost"]), "cease-dylan"),
    [{ slug: "sale-chris", name: "Chris Sale" }, { slug: "ohtani-shohei", name: "Shohei Ohtani" }]);
  assert.deepEqual(compareChoices(index, new Set(["cease-dylan"]), "cease-dylan"), []);
});

test("debug switches parse from the query; lighter graphics draw at pixel ratio 1", () => {
  assert.deepEqual(debugList("?debug=slow,nowebgl&x=1"), ["slow", "nowebgl"]);
  assert.deepEqual(debugList(""), []);
  assert.equal(pixelRatio(2, true), 1);
  assert.equal(pixelRatio(2, false), 1.5);
});

test("auto lighter graphics: median over 28 ms trips it, but never when light already or the user ever chose", () => {
  const slow = Array(40).fill(40), fine = Array(40).fill(16);
  assert.equal(shouldAutoLite(slow), true);
  assert.equal(shouldAutoLite(fine), false);
  assert.equal(shouldAutoLite([...Array(30).fill(16), ...Array(10).fill(200)]), false, "a few hitches do not move the median");
  assert.equal(shouldAutoLite(slow, { lite: true }), false);
  assert.equal(shouldAutoLite(slow, { userChose: true }), false, "an explicit choice (even 'off') is final");
  assert.equal(shouldAutoLite(Array(3).fill(90)), false, "too few frames to judge");
});

test("auto lighter graphics watcher: decides once after 3 s of active playback; idle frames and stalls do not count", () => {
  const run = (d, step, n, t0 = 0, active = true) => { let r = false; for (let i = 0; i < n; i++) r = d.frame(t0 + i * step, active) || r; return r; };
  assert.equal(run(makeAutoLite({}), 50, 80), true, "20 fps for 4 s");
  assert.equal(run(makeAutoLite({}), 50, 40), false, "only 2 s so far");
  assert.equal(run(makeAutoLite({}), 1000 / 60, 300), false);
  assert.equal(run(makeAutoLite({}), 50, 80, 0, false), false, "idle frames are ignored");
  const d = makeAutoLite({});
  assert.equal(run(d, 1000 / 60, 300), false);
  assert.equal(run(d, 50, 80, 10000), false, "already decided: never asks again");
  assert.equal(run(makeAutoLite(() => ({ userChose: true })), 50, 80), false, "options are read when it decides");
});
