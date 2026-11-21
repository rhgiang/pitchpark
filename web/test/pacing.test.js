import { test } from "node:test";
import assert from "node:assert/strict";
import { makeFrameCap, shouldDraw, MAX_FPS, IDLE_FPS, MAX_DPR, pixelRatio } from "../src/pacing.js";

/** Gaps (ms) between the frames drawn in 10 s of a `hz` display (with +-0.3 ms rAF jitter), after the cap. */
function gaps(hz) {
  const due = makeFrameCap(), out = [];
  let last = null;
  for (let k = 0; k < hz * 10; k++) {
    const now = k * 1000 / hz + 0.3 * Math.sin(k * 7.1);
    if (due(now)) { if (last !== null) out.push(now - last); last = now; }
  }
  return out.slice(hz); // after the first second (the refresh estimate has settled)
}

test("the frame cap keeps ~60 fps on fast displays with EVEN frame gaps (no 3-2 judder at 144 Hz)", () => {
  assert.equal(MAX_FPS, 60);
  for (const [hz, fps] of [[60, 60], [120, 60], [144, 72], [165, 55], [240, 60]]) {
    const g = gaps(hz), mean = g.reduce((a, b) => a + b) / g.length;
    assert.ok(Math.abs(1000 / mean - fps) < 1, `${hz} Hz -> ${(1000 / mean).toFixed(1)} fps`);
    assert.ok(Math.max(...g) - Math.min(...g) < 1.5, `${hz} Hz gaps ${Math.min(...g).toFixed(1)}..${Math.max(...g).toFixed(1)} ms`);
  }
});

test("after a stall the cap keeps drawing at its rate instead of bursting to catch up", () => {
  const due = makeFrameCap();
  for (let k = 0; k < 60; k++) due(k * 1000 / 144);
  const after = [5000, 5000 + 1000 / 144, 5000 + 2000 / 144, 5000 + 3000 / 144].filter(due).length;
  assert.equal(after, 2); // every 2nd refresh at 144 Hz
});

test("a still frame is not redrawn, except at the idle rate for the park's flags and waterfall", () => {
  assert.equal(shouldDraw(true, 1000, 999), true);
  assert.equal(shouldDraw(false, 1000, 990), false);
  assert.equal(shouldDraw(false, 1000, 1000 - 1000 / IDLE_FPS), true);
  assert.ok(IDLE_FPS < MAX_FPS / 2);
});

test("device pixel ratio is capped at 1.5", () => {
  assert.equal(MAX_DPR, 1.5);
  assert.equal(pixelRatio(1), 1);
  assert.equal(pixelRatio(1.25), 1.25);
  assert.equal(pixelRatio(2), 1.5);
});
