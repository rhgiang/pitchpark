// Eye-test page helpers (compare.js). Inputs are SYNTHETIC numbers, not real clips.
import { test } from "node:test";
import assert from "node:assert/strict";
import { clipKey, replicaTime, videoFrame, projectPoint, panAt, phaseAt, jointsAt } from "../src/compare.js";

test("clipKey: <game_pk>_<ab>_<pitch>.mp4, with or without a folder; anything else is null", () => {
  assert.equal(clipKey("824442_12_3.mp4"), "824442_12_3.mp4");
  assert.equal(clipKey("C:\\fakepath\\824442_12_3.mp4"), "824442_12_3.mp4");
  assert.equal(clipKey("824442_12_3 (1).mp4"), null);
  assert.equal(clipKey("clip.mp4"), null);
  assert.equal(clipKey("824442_12_3.mov"), null);
});

test("replicaTime: 0 at the release frame", () => {
  assert.equal(replicaTime(1.0, 30, 30), 0);
  assert.ok(Math.abs(replicaTime(0.5, 30, 30) + 0.5) < 1e-12);
});

test("videoFrame: frame f is shown from f/fps until (f+1)/fps; a frame offset shifts it", () => {
  assert.equal(videoFrame(0, 60), 0);
  assert.equal(videoFrame(10.5 / 60, 60), 10);
  assert.equal(videoFrame(11 / 60, 60), 11); // exactly on a frame start (float error tolerated)
  assert.equal(videoFrame(10.5 / 60, 60, 2), 12);
});

test("projectPoint: image = A p + b", () => {
  const A = [[2, 0, 1], [0, -3, 0]];
  assert.deepEqual(projectPoint(A, [10, 20], [1, 2, 3]), [15, 14]);
});

test("panAt: the pan for a frame, null outside the fitted range or where the frame was not fit", () => {
  const clip = { frame0: 5, b: [[1, 1], null, [3, 3]] };
  assert.deepEqual(panAt(clip, 5), [1, 1]);
  assert.equal(panAt(clip, 6), null);
  assert.deepEqual(panAt(clip, 7), [3, 3]);
  assert.equal(panAt(clip, 4), null);
  assert.equal(panAt(clip, 8), null);
});

test("phaseAt: phases are (t0, t1]; release (t = 0) closes the arm action; outside the window is null", () => {
  const phases = [
    { name: "a", t0_s: -2, t1_s: -1 }, { name: "b", t0_s: -1, t1_s: 0 }, { name: "c", t0_s: 0, t1_s: 0.5 },
  ];
  assert.equal(phaseAt(phases, -2), "a");
  assert.equal(phaseAt(phases, -1), "a");
  assert.equal(phaseAt(phases, -0.5), "b");
  assert.equal(phaseAt(phases, 0), "b");
  assert.equal(phaseAt(phases, 0.01), "c");
  assert.equal(phaseAt(phases, -2.5), null);
  assert.equal(phaseAt(phases, 0.6), null);
});

test("jointsAt: delivery joints at t s from release, linear between frames, null outside", () => {
  const d = { fps: 10, release_index: 1, joints_ft: [[[0, 0, 0], [1, 1, 1]], [[1, 0, 0], [2, 1, 1]], [[3, 0, 0], [4, 1, 1]]] };
  assert.deepEqual(jointsAt(d, 0, [0, 1]), [[1, 0, 0], [2, 1, 1]]);
  assert.deepEqual(jointsAt(d, 0.05, [0]), [[2, 0, 0]]);
  assert.equal(jointsAt(d, -0.2, [0]), null);
  assert.equal(jointsAt(d, 0.11, [0]), null);
});
