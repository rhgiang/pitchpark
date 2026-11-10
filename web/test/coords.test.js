import { test } from "node:test";
import assert from "node:assert/strict";
import { Matrix3 } from "three";
import { toScene } from "../src/coords.js";

test("toScene maps Statcast (x, y, z) to scene (x, z, -y)", () => {
  const v = toScene([1, 2, 3]);
  assert.deepEqual([v.x, v.y, v.z], [1, 3, -2]);
});

test("the mapping keeps handedness (basis determinant +1)", () => {
  const cols = [[1, 0, 0], [0, 1, 0], [0, 0, 1]].map((e) => toScene(e));
  const m = new Matrix3().set(
    cols[0].x, cols[1].x, cols[2].x,
    cols[0].y, cols[1].y, cols[2].y,
    cols[0].z, cols[1].z, cols[2].z,
  );
  assert.equal(m.determinant(), 1);
});
