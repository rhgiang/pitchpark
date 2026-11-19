import { test } from "node:test";
import assert from "node:assert/strict";
import { planFans, fanLod, fanRecords, NEAR_FT } from "../src/park/crowd.js";
import { personGeometry, farArmsGeometry, STYLE, MODE, PART } from "../src/park/person.js";

// Synthetic stand (test data, not Statcast): 4 sections x 14 seats x 12 rows on each side of home, lower deck.
const seats = [];
for (const sx of [-1, 1]) for (let row = 0; row < 12; row++) for (let i = 0; i < 56; i++) {
  seats.push({ p: [sx * (20 + i * 2), -(60 + row * 3)], h: row, yaw: 0, deck: "lower", row, rows: 12, sec: Math.floor(i / 14) });
}

test("v2 fans: same team/standing as the old crowd (old=fans is untouched), plus style and arm mode", () => {
  const a = planFans(seats, 11), b = planFans(seats, 11, { v2: true });
  a.forEach((f, i) => { assert.equal(!!f, !!b[i]); if (f) { assert.equal(f.team, b[i].team); assert.equal(f.standing, b[i].standing); } });
  assert.ok(a.every((f) => !f || f.mode === undefined)); // old plan has none
  assert.deepEqual(planFans(seats, 11, { v2: true }), b); // deterministic
});

test("v2 fans: varied skin, hair and styles; mostly resting arms, some clapping, a few cheering; standing fans' arms move", () => {
  const f = planFans(seats, 11, { v2: true }).filter(Boolean), n = f.length;
  const frac = (p) => f.filter(p).length / n;
  assert.ok(new Set(f.map((x) => x.skin)).size >= 7 && new Set(f.map((x) => x.hair)).size >= 6);
  assert.ok(frac((x) => x.style === STYLE.CAP) > 0.25 && frac((x) => x.style === STYLE.LONG) > 0.15);
  assert.ok(frac((x) => x.style === STYLE.BALD) > 0.02 && frac((x) => x.style === STYLE.BALD) < 0.12);
  assert.ok(frac((x) => x.mode === MODE.REST) > 0.55);
  assert.ok(frac((x) => x.mode === MODE.CLAP) > 0.1 && frac((x) => x.mode === MODE.CHEER) > 0.01);
  assert.ok(f.filter((x) => x.standing).every((x) => x.mode !== MODE.REST));
  // Shirts: the team colour still dominates a home fan's wardrobe (about 80%, not a uniform).
  const home = f.filter((x) => x.team === "home"), team = [0xe5604b, 0xc9402f, 0xf4ead2];
  const share = home.filter((x) => team.includes(x.shirt)).length / home.length;
  assert.ok(share > 0.6 && share < 0.95, `team shirt share ${share}`);
});

test("LOD: detailed behind home, simple far away and in the bleachers", () => {
  assert.equal(fanLod({ p: [0, -90], deck: "lower" }), "near");
  assert.equal(fanLod({ p: [0, -NEAR_FT - 5], deck: "upper" }), "far");
  assert.equal(fanLod({ p: [0, -90], deck: "bleacher" }), "far");
  assert.equal(fanLod({ p: [250, 200], deck: "lower" }), "far");
});

test("fan records: one per planned fan, standing hips high, kids smaller, deterministic", () => {
  const plan = planFans(seats, 11, { v2: true });
  const people = seats.map((s, i) => [s, plan[i]]).filter(([, f]) => f);
  const rng = (n) => { let a = n; return () => (a = (a * 16807) % 2147483647) / 2147483647; };
  const recs = fanRecords(people, rng(3));
  assert.equal(recs.length, people.length);
  assert.deepEqual(fanRecords(people, rng(3)).map((r) => [r.x, r.k, r.amp, r.ph]), recs.map((r) => [r.x, r.k, r.amp, r.ph]));
  const st = recs.find((r) => r.standing), sit = recs.find((r) => !r.standing);
  assert.ok(st.hip(false) - st.hip(true) > 1 && sit.hip(false) < 3);
  assert.ok(recs.every((r) => r.k > 0.7 && r.k < 1.1 && r.ph >= 0 && r.ph < 1));
});

test("person geometry: within the triangle budget, no NaN, parts and arm sides tagged", () => {
  const tris = (g) => g.index.count / 3;
  const near = personGeometry("near"), far = personGeometry("far"), legs = personGeometry("near", { legs: true });
  assert.ok(tris(near) <= 210 /* +26 for the toy-style arms */ && tris(far) <= 45 && tris(farArmsGeometry()) <= 56, `${tris(near)} ${tris(far)}`);
  assert.ok(tris(legs) > tris(near));
  for (const g of [near, far, legs, farArmsGeometry()]) {
    assert.ok([...g.attributes.position.array, ...g.attributes.normal.array].every(Number.isFinite));
    assert.equal(g.attributes.aPart.count, g.attributes.position.count);
  }
  const parts = new Set(near.attributes.aPart.array.filter((_, i) => i % 2 === 0));
  for (const p of [PART.SKIN, PART.SHIRT, PART.HAIR, PART.LONG, PART.BRIM, PART.EYE]) assert.ok(parts.has(p), `part ${p}`);
  const sides = new Set(near.attributes.aPart.array.filter((_, i) => i % 2 === 1));
  assert.deepEqual([...sides].sort(), [-1, 0, 1]);
  // The face looks toward -z: eyes sit in front of the head's centre (z < 0), long hair behind it.
  const pos = near.attributes.position.array, ap = near.attributes.aPart.array;
  const zs = (part) => [...Array(ap.length / 2).keys()].filter((i) => ap[2 * i] === part).map((i) => pos[3 * i + 2]);
  assert.ok(Math.max(...zs(PART.EYE)) < 0 && Math.min(...zs(PART.LONG)) > 0);
});
