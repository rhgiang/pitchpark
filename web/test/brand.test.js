import { test } from "node:test";
import assert from "node:assert/strict";
import { brandSigns, wordmarkTexture, WORDMARK_PX, PARK_NAME } from "../src/park/brand.js";
import { tvView } from "../src/cameras.js";

const plan = { kiosks: [{ p: [300, 250], yaw: 1 }, { p: [-300, 250], yaw: 2 }], gate: { p: [0, -206] }, lot: { c: [392, -40], w: 120, h: 76 } };
const signs = brandSigns(plan);

// Distance from a point to a segment, in plan.
function distSeg(p, a, b) {
  const t = Math.max(0, Math.min(1, ((p[0] - a[0]) * (b[0] - a[0]) + (p[1] - a[1]) * (b[1] - a[1])) / ((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)));
  return Math.hypot(p[0] - a[0] - t * (b[0] - a[0]), p[1] - a[1] - t * (b[1] - a[1]));
}

test("every sign stays off the sight lines: TV camera to plate, umpire to mound, centre-field eye", () => {
  const tv = tvView().position;
  const lines = [[[tv[0], tv[1]], [0, 0]], [[0, -5], [0, 60]]]; // plan segments the cameras look along
  for (const s of signs) {
    const reach = s.w / 2; // a sign's plan footprint is within half its width of its centre
    for (const [a, b] of lines) assert.ok(distSeg(s.p, a, b) > reach + 8, `${s.id} too near a sight line`);
    const bearing = Math.abs((Math.atan2(s.p[0], s.p[1]) * 180) / Math.PI);
    assert.ok(s.p[1] < 100 || bearing > 12, `${s.id} in front of the batter's eye`);
  }
});

test("signs are 4:1 boards with unit normals, and the TV backdrop and rooftop signs exist", () => {
  for (const s of signs) {
    assert.ok(Math.abs(s.w / s.h - 4) < 1e-9, s.id);
    assert.ok(Math.abs(Math.hypot(...s.n) - 1) < 1e-9, s.id);
  }
  const wall = signs.filter((s) => s.id.startsWith("backstop"));
  assert.equal(wall.length, 9);
  for (const s of wall) assert.ok(s.n[1] > 0.9 || Math.abs(s.n[0]) > 0.3, "faces home"), assert.ok(s.z < 4.5);
  assert.ok(signs.some((s) => s.roof && s.z > 93));
});

test("wordmark texture: requested size, drawn once per size, name is PITCH PARK", () => {
  assert.equal(PARK_NAME, "PITCH PARK");
  const made = [];
  const doc = { createElement: () => { const c = { getContext: () => new Proxy({ measureText: () => ({ width: 300 }) }, { get: (t, k) => t[k] ?? (() => {}), set: () => true }) }; made.push(c); return c; } };
  const t1 = wordmarkTexture(WORDMARK_PX.w, WORDMARK_PX.h, doc);
  assert.deepEqual([made[0].width, made[0].height], [1024, 256]);
  assert.equal(wordmarkTexture(WORDMARK_PX.w, WORDMARK_PX.h, doc), t1);
  assert.equal(made.length, 1);
});
