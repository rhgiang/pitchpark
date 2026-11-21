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

// The sculpted body (scripts/blender/make_pitcher.py): skinned to the skeleton's bones by name, every
// vertex fully weighted, the viewer's colours, the rig's height, and arms ending at the wrist joints
// (|x| = 2.52 ft) where grip.glb / mitt.glb take over.
const BODY = new URL("../public/models/pitcher.glb", import.meta.url);

test("pitcher.glb is a skinned rest-pose body that stops at the wrists", { skip: !existsSync(BODY) }, async () => {
  const parts = toParts((await parse(readFileSync(BODY))).scene);
  assert.ok(parts?.length, "no meshes");
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  let tris = 0;
  for (const [key, g] of parts) {
    assert.ok(["jersey", "pants", "navy", "skin", "hair", "shoe", "capDark", "rope"].includes(key), `unexpected material ${key}`);
    const p = g.attributes.position.array, si = g.attributes.skinIndex?.array, sw = g.attributes.skinWeight?.array;
    assert.ok(si && sw, `${key}: not skinned`);
    for (let i = 0; i < p.length / 3; i++) {
      for (let c = 0; c < 3; c++) { lo[c] = Math.min(lo[c], p[3 * i + c]); hi[c] = Math.max(hi[c], p[3 * i + c]); }
      const w = sw[4 * i] + sw[4 * i + 1] + sw[4 * i + 2] + sw[4 * i + 3];
      assert.ok(Math.abs(w - 1) < 0.01, `${key}: weights sum to ${w}`);
      for (let k = 0; k < 4; k++) assert.ok(si[4 * i + k] < 24, "bone index out of range");
    }
    tris += g.index.count / 3;
  }
  assert.ok(tris < 25000, `${tris} triangles`);
  assert.ok(Math.abs(hi[1] - 6.25) < 0.15 && Math.abs(lo[1]) < 0.05, `feet ${lo[1]} to cap ${hi[1]} ft`);
  assert.ok(hi[0] < 2.53 && lo[0] > -2.53 && hi[0] > 2.45, `arms end at the wrists: x ${lo[0]}..${hi[0]}`);
});

// The relaxed throwing hand shown after release (make_grip.py --open): skin only, within budget, in the
// same wrist frame as grip.glb (from just behind the wrist out along +x, a hand's length).
const OPEN = new URL("../public/models/open.glb", import.meta.url);

test("open.glb is a relaxed hand in the grip slot's wrist frame", { skip: !existsSync(OPEN) }, async () => {
  const parts = toParts((await parse(readFileSync(OPEN))).scene);
  assert.ok(parts?.length, "no meshes");
  let tris = 0, lo = Infinity, hi = -Infinity;
  for (const [key, g] of parts) {
    assert.equal(key, "skin");
    const p = g.attributes.position.array;
    for (let i = 0; i < p.length; i += 3) { lo = Math.min(lo, p[i]); hi = Math.max(hi, p[i]); }
    tris += g.index.count / 3;
  }
  assert.ok(tris <= 6000, `${tris} triangles`);
  assert.ok(lo < 0 && lo > -0.25, `forearm stub from x = ${lo}`);
  assert.ok(hi > 0.5 && hi < 0.8, `fingertips at x = ${hi} ft (a hand is ~0.6-0.7 ft from the wrist)`);
});

test("showOpenHand swaps the grip for the open hand, and leaves a figure without one alone", async () => {
  const THREE = await import("three");
  const { showOpenHand } = await import("../src/figure.js");
  const mats = ["jersey", "grip", "open"].map((name) => new THREE.MeshBasicMaterial({ name, visible: name !== "open" }));
  const fig = new THREE.Group().add(new THREE.Mesh(new THREE.BufferGeometry(), mats));
  showOpenHand(fig, true);
  assert.deepEqual(mats.map((m) => m.visible), [true, false, true]);
  showOpenHand(fig, false);
  assert.deepEqual(mats.map((m) => m.visible), [true, true, false]);
  const lone = [new THREE.MeshBasicMaterial({ name: "grip" })];
  showOpenHand(new THREE.Group().add(new THREE.Mesh(new THREE.BufferGeometry(), lone)), true);
  assert.equal(lone[0].visible, true, "no open hand loaded: keep the grip");
});
