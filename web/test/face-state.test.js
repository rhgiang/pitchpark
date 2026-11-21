import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import {
  initialState, reduce, currentAtBat, currentPitch, shownPaths, flightEnd, landed, canThrow, ballPath, PACES,
} from "../src/state.js";
import { expandPitcher } from "../src/trajectory.js";

// SYNTHETIC fixtures (both labelled synthetic in their meta): state logic only, never real results.
const replayData = expandPitcher(JSON.parse(readFileSync(new URL("fixtures/synthetic.json", import.meta.url))));
const face = JSON.parse(readFileSync(new URL("fixtures/face-synthetic.json", import.meta.url)));
const run = (s, ...actions) => actions.reduce(reduce, s);
const replay = () => run(initialState(), { type: "pickPitcher", slug: "synthetic", data: replayData });
const inFace = (extra = {}) => run(replay(), { type: "enterFace", face, seed: 7, ...extra });

test("enterFace: kind is face and there is no pitch until facePitch", () => {
  const s = inFace();
  assert.equal(s.kind, "face");
  assert.equal(currentPitch(s), null);
  assert.equal(currentAtBat(s).generated, true);
  assert.deepEqual(shownPaths(s), []);
  assert.deepEqual(s.face.opts, { stand: "R", batter: "avg", call: false, startCount: "0-0", paceS: PACES.real });
});

test("after facePitch, shownPaths is the generated sim path, in every mode", () => {
  const s = run(inFace(), { type: "facePitch" });
  const p = currentPitch(s);
  assert.equal(p.generated, true);
  assert.equal(shownPaths(s).length, 1);
  assert.equal(shownPaths(s)[0], p.sim);
  assert.equal(ballPath(s, p), p.sim);
  assert.ok(flightEnd(s) > 0.3 && flightEnd(s) < 0.6); // a real flight time, from expanded knots
  assert.equal(s.t, 0); // no delivery in the fixture: the pitch starts at release
  assert.equal(s.playing, true);
  for (const mode of ["real", "sim", "ghost"]) {
    const m = run(s, { type: "setMode", mode });
    assert.equal(m.mode, "real", "mode actions are ignored in Face him");
    assert.equal(shownPaths(m)[0], p.sim);
  }
});

test("switching back to replay restores ab, pitch and t exactly", () => {
  const base = { ...replay(), abIndex: 3, pitchIndex: 2, t: 0.1, view: "side" };
  const s = run(base, { type: "enterFace", face, seed: 1 }, { type: "facePitch" }, { type: "tick", dt: 0.5 },
    { type: "facePitch" });
  assert.equal(s.kind, "face");
  // seed 1: the first pitch is put in play, which ends the at-bat, so the second throw is ignored
  assert.equal(s.face.ab.pitches.length, 1);
  assert.equal(s.face.ab.event, "in_play");
  const back = run(s, { type: "leaveFace" });
  assert.equal(back.kind, "replay");
  assert.equal(back.face, null);
  assert.equal(back.abIndex, 3);
  assert.equal(back.pitchIndex, 2);
  assert.equal(back.t, 0.1);
  assert.equal(back.view, "side");
});

test("a stand change mid-at-bat starts a fresh at-bat at the start count", () => {
  let s = run(inFace({ opts: { startCount: "1-1" } }), { type: "facePitch" });
  const id = currentAtBat(s).id;
  assert.equal(currentAtBat(s).pitches.length, 1);
  s = run(s, { type: "faceOpts", stand: "L" });
  const ab = currentAtBat(s);
  assert.equal(ab.stand, "L");
  assert.equal(ab.pitches.length, 0);
  assert.deepEqual([ab.balls, ab.strikes], [1, 1]);
  assert.notEqual(ab.id, id, "unique id per at-bat");
  assert.equal(currentPitch(s), null);
  // an option that does not change the at-bat leaves it alone
  const same = run(s, { type: "facePitch" });
  assert.equal(run(same, { type: "faceOpts", call: true }).face.ab, same.face.ab);
  assert.equal(run(same, { type: "faceOpts", call: true }).face.opts.call, true);
  assert.equal(run(same, { type: "faceOpts", paceS: PACES.fast }).face.opts.paceS, 2);
});

test("Real/Sim/Ghost, pick at-bat and pitch navigation are ignored while in Face him", () => {
  const s = run(inFace(), { type: "facePitch" });
  for (const a of [{ type: "setMode", mode: "sim" }, { type: "setMode", mode: "ghost" }, { type: "pickAtBat", index: 0 },
    { type: "goPitch", index: 0 }, { type: "nextPitch" }, { type: "prevPitch" }]) assert.equal(reduce(s, a), s, a.type);
});

test("canThrow: only after the last pitch landed and while the at-bat is live; faceNewAtBat resets", () => {
  let s = inFace();
  assert.equal(canThrow(s), true);
  s = run(s, { type: "facePitch" });
  assert.equal(canThrow(s), false, "in flight");
  assert.equal(reduce(s, { type: "facePitch" }), s, "a throw while in flight is ignored");
  s = run(s, { type: "tick", dt: 5 });
  assert.equal(landed(s), true);
  assert.equal(canThrow(s), currentAtBat(s).event === null);
  // play it out
  for (let i = 0; i < 40 && currentAtBat(s).event === null; i++) s = run(s, { type: "facePitch" }, { type: "tick", dt: 5 });
  assert.notEqual(currentAtBat(s).event, null, "an at-bat ends within a few dozen pitches");
  assert.equal(canThrow(s), false);
  assert.equal(reduce(s, { type: "facePitch" }), s);
  const n = run(s, { type: "faceNewAtBat" });
  assert.equal(currentAtBat(n).pitches.length, 0);
  assert.equal(currentAtBat(n).event, null);
  assert.equal(canThrow(n), true);
});

test("same seed, same at-bat", () => {
  const play = () => {
    let s = inFace();
    for (let i = 0; i < 5 && currentAtBat(s).event === null; i++) s = run(s, { type: "facePitch" }, { type: "tick", dt: 5 });
    return currentAtBat(s).pitches.map((p) => `${p.pitch_type}:${p.description}:${p.plate_sim.join(",")}`);
  };
  assert.deepEqual(play(), play());
});

test("replay stays untouched: kind replay, shownPaths for a real pitch as before", () => {
  const s = run(replay(), { type: "pickAtBat", index: 0 });
  assert.equal(s.kind, "replay");
  assert.equal(s.face, null);
  assert.equal(ballPath(s, currentPitch(s)), currentPitch(s).real);
  assert.equal(reduce(s, { type: "leaveFace" }), s);
});

test("faceFromUrl: ?face=1 deep link, unknown values ignored", async () => {
  const { faceFromUrl } = await import("../src/menu.js");
  assert.equal(faceFromUrl("?pitcher=sale-chris"), null);
  assert.equal(faceFromUrl("?face=0"), null);
  assert.deepEqual(faceFromUrl("?face=1"), {});
  assert.deepEqual(faceFromUrl("?face=1&stand=L&batter=none&call=1"), { stand: "L", batter: "none", call: true });
  assert.deepEqual(faceFromUrl("?face=1&stand=X&batter=pro&call=2"), {});
});
