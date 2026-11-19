import test from "node:test";
import assert from "node:assert/strict";
import { isOld } from "../src/park/old.js";

test("old= matches names exactly, from a comma list", () => {
  assert.equal(isOld("night", "?old=night,field"), true);
  assert.equal(isOld("field", "?old=night,field"), true);
  assert.equal(isOld("field", "?old=nightfield"), false);
  assert.equal(isOld("night", "?pitcher=x"), false);
  assert.equal(isOld("night", ""), false);
  assert.equal(isOld("night"), false); // no location under node
});
