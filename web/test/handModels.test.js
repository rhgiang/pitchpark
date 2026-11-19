import { test } from "node:test";
import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { toParts } from "../src/handModels.js";

// The sculpted glove (scripts/blender/make_hands.py builds it) must load into the figure's parts:
// known material keys, finite positions, the triangle budget, and a glove-sized hand frame.
const MITT = new URL("../public/models/mitt.glb", import.meta.url);

const parse = (buf) => new Promise((ok, fail) =>
  new GLTFLoader().parse(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength), "", ok, fail));

test("mitt.glb loads into handModels parts within budget and in the hand frame", { skip: !existsSync(MITT) }, async () => {
  const parts = toParts((await parse(readFileSync(MITT))).scene);
  assert.ok(parts?.length, "no meshes");
  let tris = 0;
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  for (const [key, g] of parts) {
    assert.ok(["glove", "gloveDark", "gloveLight"].includes(key), `unexpected material ${key}`);
    const p = g.attributes.position.array;
    for (let i = 0; i < p.length; i++) {
      assert.ok(Number.isFinite(p[i]), "non-finite position");
      lo[i % 3] = Math.min(lo[i % 3], p[i]);
      hi[i % 3] = Math.max(hi[i % 3], p[i]);
    }
    tris += g.index.count / 3;
  }
  assert.ok(tris <= 8000, `${tris} triangles > 8000`);
  assert.ok(hi[0] - lo[0] > 0.85 && hi[0] - lo[0] < 1.05, `heel to tips ${hi[0] - lo[0]} ft`); // ~0.95 ft
  assert.ok(lo[0] > -0.1 && lo[0] < 0.05, "heel near the wrist (x = 0)");
  assert.ok(hi[1] < 0.15, "back of the glove just behind the hand (+y)");
  assert.ok(hi[2] > 0.3, "thumb side is +z");
});
