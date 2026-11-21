import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { isStrike, throwPitch, newAtBat, rng } from "../src/face/engine.js";
import { scoreCall, scoreType, tally, whyText, resultLine } from "../src/face/guess.js";

// SYNTHETIC fixture: scoring tests only, never real results.
const face = JSON.parse(readFileSync(new URL("./fixtures/face-synthetic.json", import.meta.url)));
const HALF_W = 17 / 24 + 1.45 / 12; // Task 3: plate half-width + ball radius
const mid = (face.zone.R.top_ft + face.zone.R.bot_ft) / 2;
const at = (x, z, extra = {}) => ({ plate_sim: [x, z, 0], pitch_type: "FF", stand: "R", ...extra });
const ctx = { face, stand: "R" };

test("scoreCall: right and wrong against the zone call", () => {
  assert.equal(scoreCall("strike", at(0, mid), ctx), "right");
  assert.equal(scoreCall("ball", at(0, mid), ctx), "wrong");
  assert.equal(scoreCall("ball", at(2, mid), ctx), "right");
  assert.equal(scoreCall("strike", at(2, mid), ctx), "wrong");
});

test("scoreCall uses the zone, not the hitter outcome (a swung-at ball is still a ball)", () => {
  assert.equal(scoreCall("ball", at(2, mid, { outcome: "whiff" }), ctx), "right");
  assert.equal(scoreCall("strike", at(0, mid, { outcome: "in_play" }), ctx), "right");
});

test("no call scores as skipped, and tally totals exclude it", () => {
  assert.equal(scoreCall(null, at(0, mid), ctx), "skipped");
  assert.equal(scoreType(null, at(0, mid)), "skipped");
  const t = tally([
    { call: "right", type: "wrong" }, { call: "skipped", type: "right" }, { call: "wrong", type: "skipped" },
  ]);
  assert.deepEqual(t, { calls: [1, 2], types: [1, 2] });
  assert.deepEqual(tally([]), { calls: [0, 0], types: [0, 0] });
});

test("a pitch on the zone edge (Task 3's half-width exactly) is a strike", () => {
  assert.equal(scoreCall("strike", at(HALF_W, mid), ctx), "right");
  assert.equal(scoreCall("ball", at(HALF_W + 1e-6, mid), ctx), "right");
  assert.equal(isStrike(face, "R", [HALF_W, mid]), true); // same rule as the engine
});

test("scoreType compares the pitch type code", () => {
  assert.equal(scoreType("FF", at(0, mid)), "right");
  assert.equal(scoreType("SL", at(0, mid)), "wrong");
});

test("whyText: '1-2 vs RHB: 41% SL'", () => {
  assert.equal(whyText({ pitch_type: "SL", why: { p: 0.412, count: "1-2", stand: "R" } }), "1-2 vs RHB: 41% SL");
  assert.equal(whyText({ pitch_type: "FF", why: { p: 0.5, count: "0-0", stand: "L" } }), "0-0 vs LHB: 50% FF");
});

test("resultLine: Statcast words, plus the at-bat ending", () => {
  assert.equal(resultLine({ description: "called_strike", events: "strikeout" }), "Called strike · Strikeout looking");
  assert.equal(resultLine({ description: "swinging_strike", events: "strikeout" }), "Swinging strike · Strikeout swinging");
  assert.equal(resultLine({ description: "foul_tip", events: "strikeout" }), "Foul tip · Strikeout swinging");
  assert.equal(resultLine({ description: "ball", events: "walk" }), "Ball · Walk");
  assert.equal(resultLine({ description: "hit_into_play", events: "in_play" }), "Hit into play · In play");
  assert.equal(resultLine({ description: "foul", events: null }), "Foul");
});

test("scoring works on a generated pitch end to end", () => {
  const { pitch } = throwPitch(face, newAtBat(face, { stand: "R" }), rng(1));
  const truth = isStrike(face, "R", pitch.plate_sim) ? "strike" : "ball";
  assert.equal(scoreCall(truth, pitch, { face, stand: "R" }), "right");
  assert.equal(scoreType(pitch.pitch_type, pitch), "right");
});
