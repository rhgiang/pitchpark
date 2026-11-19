import { test } from "node:test";
import { Vector3 } from "three";
import assert from "node:assert/strict";
import { makeBall, setSeams, seamCurve, seamCrossings, seamStyle, SEAM_AXES } from "../src/ball.js";

// The seam is the stitched curve between a baseball's two figure-8 covers.

test("seam: a closed curve on the sphere, points evenly spaced", () => {
  const pts = seamCurve(108);
  assert.equal(pts.length, 108);
  const gaps = pts.map((p, i) => Math.hypot(...p.map((v, j) => v - pts[(i + 1) % 108][j])));
  for (const p of pts) assert.ok(Math.abs(Math.hypot(...p) - 1) < 1e-9);
  assert.ok(Math.max(...gaps) / Math.min(...gaps) < 1.02, "stitch spacing uneven");
});

test("seam orientation: four-seam spin crosses 4 seams per turn, two-seam 2, never more than 4", () => {
  assert.equal(seamCrossings(SEAM_AXES[4]), 4);
  assert.equal(seamCrossings(SEAM_AXES[2]), 2);
  for (let th = 0; th <= 90; th += 10) for (let ph = 0; ph < 180; ph += 10) {
    const a = (th * Math.PI) / 180, b = (ph * Math.PI) / 180;
    const c = seamCrossings([Math.sin(a) * Math.cos(b), Math.sin(a) * Math.sin(b), Math.cos(a)], 1500);
    assert.ok(c <= 4, `${c} seams per turn at ${th}/${ph}`);
  }
});

test("seam orientation is assumed from the pitch type: sinkers two-seam, the rest four-seam", () => {
  assert.equal(seamStyle("SI"), 2);
  assert.equal(seamStyle("FF"), 4);
  assert.equal(seamStyle("SL"), 4);
  assert.equal(seamStyle(undefined), 4);
});

test("ball: 108 V stitches (216) on the seam, which turns to put the chosen axis on the spin axis", () => {
  const ball = makeBall();
  const [, seams] = ball.children;
  const stitches = seams.children.find((o) => o.isInstancedMesh);
  assert.equal(stitches.count, 216);
  for (const style of [4, 2]) {
    setSeams(ball, style);
    const p = new Vector3(...SEAM_AXES[style]).applyQuaternion(seams.quaternion); // must land on local y
    assert.ok(Math.abs(p.y - 1) < 1e-9, `style ${style}: axis -> ${p.toArray()}`);
  }
});
