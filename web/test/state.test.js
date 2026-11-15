import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import {
  initialState, reduce, currentPitch, landed, flightEnd, spinFactor, revealed,
} from "../src/state.js";
import { duration } from "../src/trajectory.js";

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

test("result is gated: not landed at release or mid-flight, landed once tick reaches the end", () => {
  let s = run(loaded(), { type: "pickAtBat", index: 0 });
  assert.equal(landed(s), false); // waiting at release
  s = run(s, { type: "replay" }, { type: "tick", dt: 0.2 });
  assert.equal(landed(s), false); // mid-flight
  s = run(s, { type: "tick", dt: 1 });
  assert.deepEqual([s.playing, landed(s)], [false, true]);
  // no_tracking has no flight: its result shows at once.
  assert.equal(landed(reduce(s, { type: "nextPitch" })), true);
});

test("a landed pitch stays landed after a mode switch (sim path may be longer than the real one)", () => {
  let s = run(loaded(), { type: "pickAtBat", index: 0 }, { type: "replay" }, { type: "tick", dt: 5 });
  assert.equal(landed(s), true);
  assert.equal(landed(reduce(s, { type: "setMode", mode: "ghost" })), true);
});

test("ghost choice survives a no_model pitch and returns on the next modelled one", () => {
  let s = run(loaded(), { type: "pickAtBat", index: 1 }, { type: "setMode", mode: "ghost" },
    { type: "nextPitch" });
  assert.equal(s.mode, "real"); // no_model
  s = reduce(s, { type: "nextPitch" });
  assert.equal(s.mode, "ghost");
});

test("prevPitch clamps at the first pitch", () => {
  const s = run(loaded(), { type: "pickAtBat", index: 0 }, { type: "prevPitch" });
  assert.equal(s.pitchIndex, 0);
});

test("tick stops at the longest shown path's end (ghost: max of real and sim)", () => {
  let s = run(loaded(), { type: "pickAtBat", index: 0 }, { type: "setMode", mode: "ghost" },
    { type: "replay" });
  const p = currentPitch(s);
  const end = Math.max(duration(p.real), duration(p.sim));
  assert.equal(flightEnd(s), end);
  s = run(s, { type: "tick", dt: 5 });
  assert.deepEqual([s.t, s.playing], [end, false]);
});

test("unknown mode or speed (e.g. from the URL) is ignored", () => {
  const s = run(loaded(), { type: "pickAtBat", index: 0 });
  assert.equal(reduce(s, { type: "setMode", mode: "bogus" }), s);
  assert.equal(reduce(s, { type: "setSpeed", speed: 7 }), s);
});

test("setView switches the view and keeps the pitch and time; unknown views are ignored", () => {
  let s = { ...loaded(), t: 0.31 };
  const view0 = s.view;
  const r = reduce(s, { type: "setView", view: "ride" });
  assert.equal(r.view, "ride");
  assert.equal(r.t, 0.31);
  assert.equal(r.pitchIndex, s.pitchIndex);
  assert.equal(reduce(s, { type: "setView", view: "nope" }).view, view0);
});

test("spin is shown slowed at 1x and at the true rate in slow motion", () => {
  assert.equal(spinFactor(1), 1 / 20);
  assert.equal(spinFactor(0.25), 1);
  assert.equal(spinFactor(0.1), 1);
});

test("default view is the umpire's", () => {
  assert.equal(initialState().view, "umpire");
  for (const v of ["umpire", "batter", "side", "ride", "free"]) {
    assert.equal(reduce(loaded(), { type: "setView", view: v }).view, v);
  }
});

test("pitch strip: goPitch jumps within the at-bat, throws from t = 0, and clamps", () => {
  let s = run(loaded(), { type: "pickAtBat", index: 0 }, { type: "replay" }, { type: "tick", dt: 0.1 });
  s = reduce(s, { type: "goPitch", index: 3 });
  assert.deepEqual([s.abIndex, s.pitchIndex, s.t, s.playing], [0, 3, 0, true]);
  assert.equal(reduce(s, { type: "goPitch", index: 99 }).pitchIndex, 3); // last pitch
  assert.equal(reduce(s, { type: "goPitch", index: -2 }).pitchIndex, 0);
});

test("pitch strip spoiler rule: a result shows only once its pitch has landed or been passed", () => {
  let s = run(loaded(), { type: "pickAtBat", index: 0 });
  assert.deepEqual([0, 1, 2, 3].map((i) => revealed(s, i)), [false, false, false, false]);
  s = run(s, { type: "replay" }, { type: "tick", dt: 0.2 });
  assert.equal(revealed(s, 0), false); // mid-flight
  s = run(s, { type: "tick", dt: 1 });
  assert.equal(revealed(s, 0), true); // landed
  s = reduce(s, { type: "goPitch", index: 2 });
  assert.deepEqual([0, 1, 2, 3].map((i) => revealed(s, i)), [true, true, false, false]);
  s = reduce(s, { type: "goPitch", index: 0 }); // going back keeps what was seen
  assert.deepEqual([0, 1, 2, 3].map((i) => revealed(s, i)), [true, true, false, false]);
  s = reduce(s, { type: "pickAtBat", index: 1 }); // a new at-bat starts hidden
  assert.equal(revealed(s, 0), false);
});

test("seek freezes playback at a time clamped to the flight", () => {
  const s = run(loaded(), { type: "pickAtBat", index: 0 }, { type: "replay" }, { type: "seek", t: 99 });
  assert.deepEqual([s.t, s.playing, landed(s)], [flightEnd(s), false, true]);
  assert.equal(revealed(s, 0), true);
});

test("no-spin toggle flips showNoSpin and survives pitch, at-bat and pitcher changes", () => {
  let s = loaded();
  assert.equal(s.showNoSpin, false);
  s = reduce(s, { type: "toggleNoSpin" });
  assert.equal(s.showNoSpin, true);
  s = run(s, { type: "nextPitch" }, { type: "pickAtBat", index: 1 },
    { type: "pickPitcher", slug: "synthetic", data });
  assert.equal(s.showNoSpin, true);
  assert.equal(reduce(s, { type: "toggleNoSpin" }).showNoSpin, false);
});

test("pitch-strip colours: Statcast descriptions map to ball, strike, foul, in play", async () => {
  const { resultKind } = await import("../src/hud.js");
  const cases = {
    ball: "b", blocked_ball: "b", hit_by_pitch: "b", called_strike: "s", swinging_strike: "s",
    swinging_strike_blocked: "s", foul_tip: "s", missed_bunt: "s", foul: "f", foul_bunt: "f",
    hit_into_play: "x",
  };
  for (const [d, k] of Object.entries(cases)) assert.equal(resultKind(d), k, d);
  assert.equal(resultKind(null), null);
});

// Delivery timeline (M5): with a delivery the clock starts pre_s before release; release stays t = 0.
const delivery = { pre_s: 2.5 };
const withDelivery = () => run(initialState(), { type: "pickPitcher", slug: "synthetic", data, delivery },
  { type: "pickAtBat", index: 0 });

test("preroll: replay starts at -pre_s with a delivery, at 0 without", () => {
  assert.equal(run(withDelivery(), { type: "replay" }).t, -2.5);
  assert.equal(run(withDelivery(), { type: "nextPitch" }).t, -2.5);
  assert.equal(run(loaded(), { type: "pickAtBat", index: 0 }, { type: "replay" }).t, 0);
});

test("seek clamps to -pre_s, and to 0 without a delivery", () => {
  assert.equal(run(withDelivery(), { type: "seek", t: -9 }).t, -2.5);
  assert.equal(run(loaded(), { type: "pickAtBat", index: 0 }, { type: "seek", t: -1 }).t, 0);
});

test("tick from -pre_s reaches release after pre_s / speed seconds", () => {
  let s = run(withDelivery(), { type: "setSpeed", speed: 0.25 }, { type: "replay" });
  for (let i = 0; i < 10; i++) s = reduce(s, { type: "tick", dt: 1 }); // 10 s x 0.25 = 2.5 s
  assert.ok(Math.abs(s.t) < 1e-9);
});

test("no spoiler during the windup", () => {
  const s = run(withDelivery(), { type: "seek", t: -0.5 });
  assert.equal(landed(s), false);
  const m = run(s, { type: "setMode", mode: "ghost" });
  assert.equal(landed(m), false);
  assert.equal(m.seen, 0);
});
