import { test } from "node:test";
import assert from "node:assert/strict";
import {
  BASES, fenceDistFt, POLE_R, POLE_L, fieldBoundary, inside, resample, polar,
} from "../src/park/layout.js";

// The park is art, except where it must be a real field (OBR 2.01, Appendix 2).
const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);

test("bases: 90 ft apart, 127 ft 3 3/8 in from home to second", () => {
  const home = [0, 0];
  for (const [a, b] of [[home, BASES.first], [BASES.first, BASES.second], [BASES.second, BASES.third], [BASES.third, home]]) {
    assert.ok(Math.abs(dist(a, b) - 90) < 1e-9);
  }
  assert.ok(Math.abs(dist(home, BASES.second) - (127 + 3.375 / 12)) < 0.01);
});

test("fence: 330 ft at the poles, 375 in the gaps, 400 to centre, falling monotonically", () => {
  assert.ok(Math.abs(fenceDistFt(0) - 400) < 1e-9);
  assert.ok(Math.abs(fenceDistFt(22.5) - 375) < 1e-9);
  assert.ok(Math.abs(fenceDistFt(-45) - 330) < 1e-9);
  for (let a = 0; a < 45; a += 0.5) assert.ok(fenceDistFt(a + 0.5) < fenceDistFt(a));
});

test("foul poles sit on the 45 deg foul lines at the fence", () => {
  for (const [p, s] of [[POLE_R, 1], [POLE_L, -1]]) {
    assert.ok(Math.abs(p[0] - s * p[1]) < 1e-9); // on x = +-y
    assert.ok(Math.abs(Math.hypot(...p) - 330) < 1e-9);
  }
});

test("field boundary encloses home, the bases, the mound and the fence's inside; not beyond it", () => {
  const poly = fieldBoundary();
  for (const p of [[0, 0.5], BASES.first.map((v, i) => v - (i ? 0 : 1)), BASES.second, [0, 60.5], polar(0, 399), polar(30, fenceDistFt(30) - 1)]) {
    assert.ok(inside(p, poly), `inside: ${p}`);
  }
  for (const p of [polar(0, 401), polar(-30, fenceDistFt(30) + 1), [0, -61]]) assert.ok(!inside(p, poly), `outside: ${p}`);
});

test("resample keeps both ends and spaces points evenly", () => {
  const pts = resample([[0, 0], [10, 0], [10, 10]], 2);
  assert.deepEqual(pts[0], [0, 0]);
  assert.deepEqual(pts.at(-1), [10, 10]);
  for (let i = 1; i < pts.length - 1; i++) assert.ok(Math.abs(dist(pts[i], pts[i - 1]) - 2) < 1e-9);
});
