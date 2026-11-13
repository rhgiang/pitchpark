import { test } from "node:test";
import assert from "node:assert/strict";
import * as THREE from "three";
import { terrain, isTerrain, TERRAIN_ID, makeToonMaterial } from "../src/look.js";
import { buildField } from "../src/field.js";
import { mergeStatic } from "../src/park/build.js";

// Ryan (2026-11-13): no ink line round the mound. Ground features (mound, plate, rubber, the
// painted ground, bullpen mounds) are terrain: the outline pre-pass writes TERRAIN_ID for them
// and the edge pass draws no ground-on-ground ink there (look.js EDGE_SHADER).

const ids = (mesh) => [...new Set(mesh.geometry.attributes.objId?.array ?? [])];

test("terrain() tags every vertex with TERRAIN_ID (negative: never a real object ID)", () => {
  const m = terrain(new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), makeToonMaterial(0xffffff)));
  assert.ok(TERRAIN_ID < 0);
  assert.deepEqual(ids(m), [TERRAIN_ID]);
  assert.equal(m.geometry.attributes.objId.count, m.geometry.attributes.position.count);
  assert.ok(isTerrain(m));
  assert.ok(!isTerrain(new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1))));
});

test("field: mound, plate and rubber are terrain (no ink against the ground)", () => {
  const field = buildField();
  const meshes = [];
  field.traverse((o) => o.isMesh && meshes.push(o));
  assert.equal(meshes.length, 3);
  for (const m of meshes) assert.deepEqual(ids(m), [TERRAIN_ID]);
});

test("mergeStatic keeps terrain as terrain and objects as distinct positive IDs", () => {
  const root = new THREE.Group();
  const dirt = makeToonMaterial(0xc9875a);
  const a = terrain(new THREE.Mesh(new THREE.CylinderGeometry(2, 8, 1), dirt));
  const b = terrain(new THREE.Mesh(new THREE.CylinderGeometry(2, 8, 1), dirt));
  b.position.x = 20;
  const c = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), dirt); // same look, but an object
  const d = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), dirt);
  d.position.x = 5;
  root.add(a, b, c, d);
  mergeStatic(root);
  const merged = root.children.filter((o) => o.isMesh);
  assert.equal(merged.length, 2, "terrain and objects merge separately");
  const t = merged.find(isTerrain), o = merged.find((m) => !isTerrain(m));
  assert.deepEqual(ids(t), [TERRAIN_ID]);
  assert.ok(ids(o).length === 2 && ids(o).every((v) => v > 0));
});
