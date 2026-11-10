import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { initialState, reduce, currentPitch } from "../src/state.js";

// Synthetic fixture (labelled synthetic in its meta); at-bat 0 = "6-1", at-bat 1 = "28-1".
const data = JSON.parse(readFileSync(new URL("fixtures/synthetic.json", import.meta.url)));
const run = (s, ...actions) => actions.reduce(reduce, s);
const loaded = () => run(initialState(), { type: "pickPitcher", slug: "synthetic", data });

test("no_model pitch: setMode sim leaves mode real and modeAllowed false", () => {
  const s = run(loaded(), { type: "pickAtBat", index: 1 }, { type: "nextPitch" },
    { type: "setMode", mode: "sim" });
  assert.equal(currentPitch(s).status, "no_model");
  assert.equal(s.mode, "real");
  assert.equal(s.modeAllowed, false);
  assert.equal(run(s, { type: "setMode", mode: "ghost" }).mode, "real");
});

test("nextPitch past a no_tracking pitch keeps the at-bat and shows that row's raw count", () => {
  let s = run(loaded(), { type: "pickAtBat", index: 0 }, { type: "nextPitch" });
  assert.equal(currentPitch(s).status, "no_tracking");
  assert.deepEqual([currentPitch(s).balls, currentPitch(s).strikes], [1, 0]);
  s = run(s, { type: "tick", dt: 0.1 }, { type: "nextPitch" });
  assert.equal(s.abIndex, 0);
  assert.equal(s.pitchIndex, 2);
  assert.deepEqual([currentPitch(s).balls, currentPitch(s).strikes], [1, 1]);
});

test("pickPitcher or pickAtBat while playing resets t, playing, pitchIndex and mode", () => {
  const mid = run(loaded(), { type: "pickAtBat", index: 0 }, { type: "nextPitch" },
    { type: "nextPitch" }, { type: "setMode", mode: "ghost" }, { type: "tick", dt: 0.1 });
  assert.equal(mid.playing, true);
  assert.equal(mid.mode, "ghost");
  assert.ok(mid.t > 0);
  for (const action of [{ type: "pickPitcher", slug: "synthetic", data },
    { type: "pickAtBat", index: 1 }]) {
    const s = reduce(mid, action);
    assert.deepEqual([s.t, s.playing, s.pitchIndex, s.mode], [0, false, 0, "real"]);
  }
});

test("slow motion: tick dt 0.1 at speed 0.25 advances t by 0.025", () => {
  const s = run(loaded(), { type: "pickAtBat", index: 0 }, { type: "setSpeed", speed: 0.25 },
    { type: "replay" }, { type: "tick", dt: 0.1 });
  assert.ok(Math.abs(s.t - 0.025) < 1e-12);
});

test("random at-bat is seeded: same seed, same at-bat", () => {
  const pick = (seed) => reduce(loaded(), { type: "pickAtBat", index: "random", seed }).abIndex;
  assert.equal(pick(7), pick(7));
  const seen = new Set(Array.from({ length: 20 }, (_, i) => pick(i)));
  assert.deepEqual([...seen].sort(), [0, 1]); // both at-bats reachable
});
