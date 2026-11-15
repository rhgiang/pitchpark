import { test } from "node:test";
import assert from "node:assert/strict";
import { frameSlot, shouldDraw, MAX_FPS, IDLE_FPS, MAX_DPR, pixelRatio } from "../src/pacing.js";

/** Frames drawn in 10 s of a `hz` display, with the 60 fps cap. */
function capped(hz) {
  let next = 0, n = 0;
  for (let now = 0; now < 10000; now += 1000 / hz) {
    const s = frameSlot(now, next);
    next = s.next;
    if (s.due) n++;
  }
  return n / 10;
}

test("the frame cap holds a 144 Hz or 120 Hz display to ~60 fps and leaves a 60 Hz one alone", () => {
  assert.equal(MAX_FPS, 60);
  for (const hz of [144, 120, 165, 240]) assert.ok(Math.abs(capped(hz) - 60) <= 1.5, `${hz} Hz -> ${capped(hz)} fps`);
  assert.ok(Math.abs(capped(60) - 60) <= 0.5, `60 Hz -> ${capped(60)} fps`);
  // A slightly early 60 Hz frame (jitter) is still due.
  assert.equal(frameSlot(16.0, 16.67).due, true);
});

test("after a stall the cap resyncs instead of bursting to catch up", () => {
  const s = frameSlot(5000, 100);
  assert.equal(s.due, true);
  assert.equal(frameSlot(5001, s.next).due, false);
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
