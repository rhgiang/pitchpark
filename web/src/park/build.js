import * as THREE from "three";
import { mergeGeometries } from "three/examples/jsm/utils/BufferGeometryUtils.js";
import { normals } from "./layout.js";
import { isTerrain, TERRAIN_ID } from "../look.js";

// Small geometry helpers shared by the park modules. Statcast (x, y) -> scene (x, -y).

const tmp = new THREE.Color();

/** Linear colour of a hex, times a shade factor (baked AO). */
export const shadeOf = (hex, k = 1) => tmp.set(hex).multiplyScalar(k).toArray();

/**
 * Sweep a 2D profile along a plan-view path: the stands, walls and roofs are all this.
 * Each profile segment is [[s0, h0], [s1, h1], hex, ao0, ao1]: s = ft outward from the path
 * (along its normals), h = ft up; ao0/ao1 darken the segment's two edges (vertex colours, so
 * the back of each seat row is darker where it meets the next riser: generated contact
 * shading). Every segment gets its own vertices, so creases stay sharp (flat normal across the
 * profile, smooth along the path).
 * @param stripe if > 0, colour alternates between hex and `stripeHex` every `stripe` path steps
 *   (the cricket park's red/white roof), and path steps get their own vertices too.
 */
export function sweep(path, profile, { stripe = 0, stripeHex = 0xffffff } = {}) {
  const n = normals(path);
  const pos = [], col = [], idx = [];
  const at = (i, [s, h]) => [path[i][0] + n[i][0] * s, h, -(path[i][1] + n[i][1] * s)];
  for (const [a, b, hex, ao0 = 1, ao1 = 1] of profile) {
    if (stripe) {
      for (let i = 0; i + 1 < path.length; i++) {
        const c = Math.floor(i / stripe) % 2 ? stripeHex : hex;
        const v = pos.length / 3;
        pos.push(...at(i, a), ...at(i + 1, a), ...at(i + 1, b), ...at(i, b));
        col.push(...shadeOf(c, ao0), ...shadeOf(c, ao0), ...shadeOf(c, ao1), ...shadeOf(c, ao1));
        idx.push(v, v + 1, v + 2, v, v + 2, v + 3);
      }
      continue;
    }
    const v0 = pos.length / 3;
    path.forEach((_, i) => {
      pos.push(...at(i, a), ...at(i, b));
      col.push(...shadeOf(hex, ao0), ...shadeOf(hex, ao1));
    });
    for (let i = 0; i + 1 < path.length; i++) {
      const v = v0 + 2 * i;
      idx.push(v, v + 2, v + 3, v, v + 3, v + 1);
    }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
  g.setAttribute("color", new THREE.Float32BufferAttribute(col, 3));
  g.setIndex(idx);
  g.computeVertexNormals();
  return g;
}

/**
 * Flat end cap for a sweep: the closed polygon `outline` ([s, h] points) standing at path point
 * i, in the plane of the normal and up.
 */
export function cap(path, i, outline, hex) {
  const n = normals(path)[i];
  const tris = THREE.ShapeUtils.triangulateShape(outline.map(([s, h]) => new THREE.Vector2(s, h)), []);
  const pos = outline.flatMap(([s, h]) => [path[i][0] + n[0] * s, h, -(path[i][1] + n[1] * s)]);
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
  g.setAttribute("color", new THREE.Float32BufferAttribute(outline.flatMap(() => shadeOf(hex)), 3));
  g.setIndex(tris.flat());
  g.computeVertexNormals();
  return g;
}

/**
 * A Group standing at Statcast [x, y] whose local x axis points along plan direction `dir`
 * and local z along its right-hand normal (dir rotated -90 deg). Children use local feet.
 */
export function frameAt([x, y], [dx, dy]) {
  const g = new THREE.Group();
  g.position.set(x, 0, -y);
  g.rotation.y = Math.atan2(dy, dx);
  return g;
}

/** frameAt for a point `angleDeg` from the centre-field line, local z pointing away from home. */
export function radialFrame(angleDeg, r) {
  const t = (angleDeg * Math.PI) / 180;
  return frameAt([r * Math.sin(t), r * Math.cos(t)], [-Math.cos(t), Math.sin(t)]);
}

/** A box mesh (w, h, d in local x, y, z) with its base centred at (x, y, z). */
export function box(material, w, h, d, x = 0, y = 0, z = 0) {
  const m = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), material);
  m.position.set(x, y + h / 2, z);
  m.castShadow = m.receiveShadow = true;
  return m;
}

/** Set an InstancedMesh's i-th matrix from position, yaw (rad, about scene y) and scale. */
const M = new THREE.Matrix4(), Q = new THREE.Quaternion(), S = new THREE.Vector3(), P = new THREE.Vector3(), UP = new THREE.Vector3(0, 1, 0);
export function place(inst, i, [x, y, z], yaw = 0, s = 1) {
  P.set(x, y, z);
  Q.setFromAxisAngle(UP, yaw);
  if (typeof s === "number") S.set(s, s, s); else S.set(...s);
  inst.setMatrixAt(i, M.compose(P, Q, S));
}

/** Canvas texture with text, for signs and wall numbers. */
export function textTexture(text, { w = 256, h = 128, bg = null, fg = "#fff", font = "bold 90px sans-serif" } = {}) {
  const c = document.createElement("canvas");
  c.width = w; c.height = h;
  const g = c.getContext("2d");
  if (bg) { g.fillStyle = bg; g.fillRect(0, 0, w, h); }
  g.fillStyle = fg; g.font = font; g.textAlign = "center"; g.textBaseline = "middle";
  g.fillText(text, w / 2, h / 2 + 4);
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  t.anisotropy = 8;
  return t;
}

/*
 * Merge the park's static meshes, one merged mesh per material look, to cut draw calls: the art
 * pass adds hundreds of small boxes (gate, kiosks, posts, shelters), and each draw costs CPU time in
 * both the outline pre-pass and the main pass whatever its size. Skipped: instanced, skinned and
 * textured meshes, other layers (not outlined / ink-proof), transparent ones, shared materials that
 * time of day recolours (look.js marks them userData.shared; they merge by identity), and anything
 * under an object with an animation tick. Each source mesh keeps its own object ID as a vertex
 * attribute (objId), which the outline pre-pass (look.js) uses, so merged objects still get their
 * outlines where they meet.
 */
export function mergeStatic(root) {
  root.updateMatrixWorld(true);
  const groups = new Map();
  const animated = new Set();
  root.traverse((o) => { if (o.userData.tick) o.traverse((c) => animated.add(c)); });
  root.traverse((o) => {
    const m = o.material;
    if (!o.isMesh || o.isInstancedMesh || o.isSkinnedMesh || animated.has(o) || o.layers.mask !== 1) return;
    if (Array.isArray(m) || m.map || m.transparent || !m.isMeshToonMaterial) return;
    const key = m.userData.shared ? m.uuid : [m.color.getHex(), m.emissive.getHex(), m.emissiveIntensity,
      m.defines?.SURFACE, m.vertexColors, m.side, o.castShadow, o.receiveShadow, isTerrain(o)].join("|");
    if (!groups.has(key)) groups.set(key, { material: m, cast: o.castShadow, receive: o.receiveShadow, list: [] });
    groups.get(key).list.push(o);
  });
  let id = 1;
  for (const { material, cast, receive, list } of groups.values()) {
    if (list.length < 2) continue;
    const geos = list.map((o) => {
      let g = o.geometry.index ? o.geometry.toNonIndexed() : o.geometry.clone();
      for (const name of Object.keys(g.attributes)) {
        if (!["position", "normal", ...(material.vertexColors ? ["color"] : [])].includes(name)) g.deleteAttribute(name);
      }
      g.applyMatrix4(o.matrixWorld);
      // Terrain keeps its shared ID (no ink between pieces of ground: look.js terrain()).
      g.setAttribute("objId", new THREE.Float32BufferAttribute(new Float32Array(g.attributes.position.count).fill(isTerrain(o) ? TERRAIN_ID : id++), 1));
      return g;
    });
    const merged = mergeGeometries(geos);
    if (!merged) continue; // attribute mismatch: leave those meshes as they are
    for (const o of list) o.removeFromParent();
    const mesh = new THREE.Mesh(merged, material);
    if (isTerrain(list[0])) mesh.userData.terrain = true;
    mesh.castShadow = cast;
    mesh.receiveShadow = receive;
    root.add(mesh);
  }
}
