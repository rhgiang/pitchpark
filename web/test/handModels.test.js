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
  // the forearm (along -x, centred on y = z = 0) enters through the heel opening: glove all round it
  assert.ok(hi[1] > 0.1 && lo[1] < -0.1, "glove on both sides of the wrist (back +y, palm -y)");
  assert.ok(hi[2] > 0.3, "thumb side is +z");
});

// The posed throwing hand (scripts/blender/make_grip.py): skin only, within budget, and holding the
// ball where the viewer draws it (centre (0.28, 0, 0) ft): skin at most 4 mm into the ball, fingers
// over its top (+y, the back of the hand), thumb side +z.
const GRIP = new URL("../public/models/grip.glb", import.meta.url);

test("grip.glb holds the ball at the hand joint, within budget", { skip: !existsSync(GRIP) }, async () => {
  const parts = toParts((await parse(readFileSync(GRIP))).scene);
  assert.ok(parts?.length, "no meshes");
  const R = 1.45 / 12; // ball.js BALL_RADIUS_FT
  let tris = 0, deepest = -Infinity;
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  for (const [key, g] of parts) {
    assert.equal(key, "skin");
    const p = g.attributes.position.array;
    for (let i = 0; i < p.length; i += 3) {
      for (let c = 0; c < 3; c++) {
        assert.ok(Number.isFinite(p[i + c]), "non-finite position");
        lo[c] = Math.min(lo[c], p[i + c]);
        hi[c] = Math.max(hi[c], p[i + c]);
      }
      deepest = Math.max(deepest, R - Math.hypot(p[i] - 0.28, p[i + 1], p[i + 2]));
    }
    tris += g.index.count / 3;
  }
  assert.ok(tris <= 6000, `${tris} triangles > 6000`);
  assert.ok(deepest < 0.004 / 0.3048, `skin ${(deepest * 304.8).toFixed(1)} mm inside the ball`);
  assert.ok(lo[0] < 0 && hi[0] > 0.28 + R, "from behind the wrist to past the ball");
  assert.ok(hi[1] > R, "fingers over the top of the ball (+y)");
  assert.ok(hi[2] > 0.1, "thumb side is +z");
});
