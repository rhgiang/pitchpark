import { test } from "node:test";
import assert from "node:assert/strict";
import { batterEye } from "../src/cameras.js";

test("a right-handed batter stands on the catcher's left (x < 0), a lefty on the right", () => {
  assert.ok(batterEye("R").position[0] < 0);
  assert.ok(batterEye("L").position[0] > 0);
});
