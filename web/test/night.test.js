import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";

// look.js needs a browser (canvas), so check the night tables as text: the new night keeps its
// floodlights drawn in and a dark ambient, and ?old=night still selects the old table.
const src = fs.readFileSync(new URL("../src/look.js", import.meta.url), "utf8");

test("night table: dark ambient, floodlights drawn in, old night kept for ?old=night", () => {
  assert.match(src, /flood: \[200, 380\]/);
  assert.match(src, /ambient: 0\.3,/);
  assert.match(src, /const OLD_NIGHT = \{/);
  assert.match(src, /name === "night" && isOld\("night"\) \? OLD_NIGHT : TIMES\[name\]/);
});
