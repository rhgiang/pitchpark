import { test } from "node:test";
import assert from "node:assert/strict";
import { TOUR_STEPS, TOUR_KEY, tourSeen, markTourSeen, clearTourSeen, shouldAutoTour, nextStep, prevStep,
  resolveTarget, placeCard, HOW, loadingText, errorText } from "../src/guide.js";

const memory = () => { const m = new Map(); return { getItem: (k) => m.get(k) ?? null, setItem: (k, v) => m.set(k, v), removeItem: (k) => m.delete(k) }; };
const broken = { getItem() { throw new Error("blocked"); }, setItem() { throw new Error("blocked"); }, removeItem() { throw new Error("blocked"); } };

test("seen flag round-trips and survives storage that throws or is missing", () => {
  const s = memory();
  assert.equal(tourSeen(s), false);
  markTourSeen(s);
  assert.equal(s.getItem(TOUR_KEY), "1");
  assert.equal(tourSeen(s), true);
  clearTourSeen(s);
  assert.equal(tourSeen(s), false);
  assert.equal(tourSeen(broken), false);
  assert.doesNotThrow(() => { markTourSeen(broken); clearTourSeen(broken); });
  assert.equal(tourSeen(null), false);
  assert.doesNotThrow(() => markTourSeen(null));
});

test("the tour auto-shows once, and never on a deep link", () => {
  assert.equal(shouldAutoTour("", false), true);
  assert.equal(shouldAutoTour("", true), false);
  assert.equal(shouldAutoTour("?pitcher=sale-chris", false), false);
});

test("four steps, stepping clamps and ends", () => {
  assert.equal(TOUR_STEPS.length, 4);
  assert.equal(nextStep(0, 4), 1);
  assert.equal(nextStep(3, 4), null);
  assert.equal(prevStep(0), 0);
  assert.equal(prevStep(2), 1);
});

test("target resolution: padded rect, hidden or missing -> null", () => {
  const r = { left: 100, top: 50, width: 40, height: 20 };
  assert.deepEqual(resolveTarget("#a", () => r, 6), { left: 94, top: 44, width: 52, height: 32 });
  assert.equal(resolveTarget("#a", () => ({ left: 0, top: 0, width: 0, height: 0 })), null);
  assert.equal(resolveTarget("#a", () => null), null);
});

test("card placement: below if it fits, else above, clamped, centred without a ring", () => {
  const card = { width: 300, height: 150 };
  assert.deepEqual(placeCard({ left: 10, top: 10, width: 100, height: 40 }, card, 1000, 800), { left: 12, top: 64 });
  assert.equal(placeCard({ left: 400, top: 700, width: 100, height: 40 }, card, 1000, 800).top, 700 - 14 - 150);
  assert.equal(placeCard({ left: 950, top: 10, width: 40, height: 40 }, card, 1000, 800).left, 1000 - 300 - 12);
  assert.deepEqual(placeCard(null, card, 1000, 800), { left: 350, top: 325 });
  assert.equal(placeCard(null, card, 200, 800).left, 12); // card shrinks to the viewport minus margins, then centres
});

test("How it works carries the sourced figures with their qualifiers", () => {
  assert.equal(HOW.steps.length, 5);
  const nums = HOW.steps.map((s) => s.n);
  assert.deepEqual(nums, ["public data", "2.0–4.0 in", "22 of 22", "6 of 9", "~2.1 in"]);
  assert.match(HOW.steps[1].c, /four-seamers/);
  assert.match(HOW.steps[1].c, /held-out/);
});

test("loading copy names the pitcher", () => {
  assert.equal(loadingText("Chris Sale"), "Loading Chris Sale's delivery…");
  assert.equal(errorText("Chris Sale"), "Couldn't load Chris Sale's delivery.");
});
