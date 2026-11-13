import { test } from "node:test";
import assert from "node:assert/strict";
import { chalkSegments, CHALK_WIDTH_FT, CHALK } from "../src/park/layout.js";

// The chalk is drawn analytically in the ground shader from these segments (surface.js), so the
// lines are sharp at any distance. Their geometry is real (OBR 2.01-2.03, Appendix 2).
const near = (a, b, tol = 1e-3) => Math.abs(a - b) < tol;
const segs = chalkSegments();
const of = (kind) => segs.filter((s) => s[2] === kind);

test("lines are 3 in wide", () => assert.ok(near(CHALK_WIDTH_FT, 0.25)));

test("foul lines run along x = +-y out to the poles at 330 ft", () => {
  const foul = of(CHALK.foul);
  assert.equal(foul.length, 2);
  for (const [a, b] of foul) {
    for (const p of [a, b]) assert.ok(near(Math.abs(p[0]), p[1]));
    assert.ok(near(Math.hypot(...b), 330));
  }
});

test("batter's boxes: 4 x 6 ft, inner line 6 in off the 17 in plate, centred on its middle", () => {
  const box = of(CHALK.box);
  assert.equal(box.length, 8);
  for (const s of [-1, 1]) {
    const pts = box.flatMap(([a, b]) => [a, b]).filter((p) => Math.sign(p[0]) === s);
    const xs = pts.map((p) => Math.abs(p[0])), ys = pts.map((p) => p[1]);
    assert.ok(near(Math.min(...xs), 8.5 / 12 + 0.5));
    assert.ok(near(Math.max(...xs) - Math.min(...xs), 4));
    assert.ok(near(Math.max(...ys) - Math.min(...ys), 6));
    assert.ok(near((Math.max(...ys) + Math.min(...ys)) / 2, 17 / 24));
  }
});

test("catcher's box: 43 in wide, 8 ft deep behind the batter's boxes", () => {
  const pts = of(CHALK.catcher).flatMap(([a, b]) => [a, b]);
  const xs = pts.map((p) => p[0]), ys = pts.map((p) => p[1]);
  assert.ok(near(Math.max(...xs) - Math.min(...xs), 43 / 12));
  assert.ok(near(Math.max(...ys) - Math.min(...ys), 8));
});
