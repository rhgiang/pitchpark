import * as THREE from "three";
import { PALETTE, makeToonMaterial, noOutline, timeOfDay } from "../look.js";
import { place } from "./build.js";
import { polar, fenceDistFt, rng } from "./layout.js";
import { fallTexture } from "./surrounds.js";
import { FALLS } from "./signature.js";

/*
 * The Falls (our McCovey Cove): beyond the right-field wall the ground ends in a cliff with a waterfall
 * pouring into a pool. Low-poly and flat-shaded, as the rest of the park:
 *  - the cliff is a polar height grid (bearing x distance behind the fence): a steep face, a plateau, a taper,
 *    cooler rock lower down and grass on top-facing triangles;
 *  - the fall is one strip hugging the face (the island waterfall's scrolling stripe texture, surrounds.js),
 *    running on over the plateau as the river;
 *  - the pool is a stone-rimmed slab with a water plane, foam rings and a few mist puffs (one instanced mesh);
 *  - boulders round the rim.
 * Statcast ft in (bearing a deg from the centre-field line, r from home plate), scene ft out.
 */

const R = rng(77);
const fenceAt = (a) => fenceDistFt(a);
const smooth = (t) => { const c = Math.min(1, Math.max(0, t)); return c * c * (3 - 2 * c); };
const toScene = ([x, y], h) => [x, h, -y];
const hash = (i, j) => { const s = Math.sin(i * 127.1 + j * 311.7) * 43758.5453; return s - Math.floor(s); };

const A_STEP = 1.5;
// Distances behind the cliff's foot where the grid has rows: dense on the face, sparse on the long back slope.
const OFFS = [-6, 0, 6, 14, 24, 34, FALLS.faceRun, 60, 78, 98, 124, 156, 196, 260];
const A_CENTRE = (FALLS.aCliff0 + FALLS.aCliff1) / 2;
const HALF_SPAN = (FALLS.aCliff1 - FALLS.aCliff0) / 2;

/** Height (ft) of the cliff at bearing a, `off` ft behind its foot; i, j are grid indices for repeatable roughness. */
export function cliffHeight(a, off, i = 0, j = 0) {
  const lateral = smooth((HALF_SPAN - Math.abs(a - A_CENTRE)) / 11);
  const notch = Math.abs(a - FALLS.aFall) <= 3.1; // the fall's columns stay smooth, and a little lower
  const top = FALLS.cliffH * lateral * (notch ? 0.94 : 0.86 + 0.24 * hash(i, 0));
  if (off <= 0) return 0;
  const rough = notch ? 0 : (hash(i, j) - 0.5) * 7 * lateral;
  if (off <= FALLS.faceRun) return top * Math.pow(off / FALLS.faceRun, 0.85) + (off > 6 ? rough * Math.min(1, off / 20) : 0);
  const plateau = top * (1 - 0.1 * Math.min(1, (off - FALLS.faceRun) / 110));
  return Math.max(0, plateau * (1 - smooth((off - 150) / 110)) + rough * 0.6);
}

/** Cliff mesh: a polar grid, flat-shaded, vertex-coloured by slope and height. */
function cliffGeometry() {
  const cols = Math.round((FALLS.aCliff1 - FALLS.aCliff0) / A_STEP);
  const P = [];
  for (let i = 0; i <= cols; i++) {
    const a = FALLS.aCliff0 + i * A_STEP;
    P.push(OFFS.map((off, j) => {
      const h = cliffHeight(a, off, i, j);
      const xy = polar(a, fenceAt(a) + FALLS.cliffOff + off);
      return [xy[0], h, -xy[1]];
    }));
  }
  const pos = [], col = [];
  const c = new THREE.Color();
  const A = new THREE.Vector3(), B = new THREE.Vector3(), C = new THREE.Vector3(), N = new THREE.Vector3();
  const tri = (p, q, r, k) => {
    A.set(...p); B.set(...q); C.set(...r);
    N.crossVectors(B.clone().sub(A), C.clone().sub(A)).normalize();
    const up = Math.abs(N.y);
    const mid = (p[1] + q[1] + r[1]) / 3;
    if (up > 0.72 && mid > 4) c.setHex(PALETTE.foliage[k % 3 + 1]).multiplyScalar(0.95 + 0.1 * hash(k, 3)); // grass on the top
    else if (up > 0.72) c.setHex(PALETTE.grassFoul);
    else c.setHex([0xa08a74, 0x8a7360, 0x76604f, 0xb5a28c][Math.floor(hash(k, 9) * 4)]).multiplyScalar(0.9 + 0.35 * Math.min(1, mid / 90));
    pos.push(...p, ...q, ...r);
    for (let n = 0; n < 3; n++) col.push(c.r, c.g, c.b);
  };
  let k = 0;
  for (let i = 0; i < cols; i++) for (let j = 0; j < OFFS.length - 1; j++) {
    const a = P[i][j], b = P[i + 1][j], d = P[i][j + 1], e = P[i + 1][j + 1];
    if (a[1] + b[1] + d[1] + e[1] < 0.5) continue;
    tri(a, d, b, k++); tri(b, d, e, k++);
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
  g.setAttribute("color", new THREE.Float32BufferAttribute(col, 3));
  g.computeVertexNormals();
  return g;
}

/** The pool outline as a plan polygon [[x, y]]: bearings a0..a1 (inset by `ia` deg), ft behind the fence poolIn..poolOut (inset by `ir`). */
function poolPolygon(ia, ir) {
  const pts = [];
  const n = 24;
  for (let k = 0; k <= n; k++) { const a = FALLS.a0 + ia + ((FALLS.a1 - FALLS.a0 - 2 * ia) * k) / n; pts.push(polar(a, fenceAt(a) + FALLS.poolIn + ir)); }
  for (let k = n; k >= 0; k--) { const a = FALLS.a0 + ia + ((FALLS.a1 - FALLS.a0 - 2 * ia) * k) / n; pts.push(polar(a, fenceAt(a) + FALLS.poolOut - ir)); }
  return pts;
}

/** A plan polygon as a Shape laid flat: (x, y_plan) -> scene (x, 0, -y_plan) once rotated by -90 deg about x. */
const shapeOf = (pts) => new THREE.Shape(pts.map(([x, y]) => new THREE.Vector2(x, y)));

/** The waterfall strip: along the face and on over the plateau, `halfW` ft either side of the centre line. */
function fallStrip(halfW, offs) {
  const pos = [], uv = [], idx = [];
  const a = FALLS.aFall;
  offs.forEach((off, k) => {
    const r = fenceAt(a) + FALLS.cliffOff + off;
    const h = cliffHeight(a, off, Math.round((a - FALLS.aCliff0) / A_STEP), 0) + 0.5;
    for (const s of [-1, 1]) {
      const aa = a + (s * (halfW / r) * 180) / Math.PI;
      pos.push(...toScene(polar(aa, r - 0.15), h));
      uv.push(s < 0 ? 0 : 1, off / 16);
    }
    if (k > 0) { const v = (k - 1) * 2; idx.push(v, v + 1, v + 3, v, v + 3, v + 2); }
  });
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
  g.setAttribute("uv", new THREE.Float32BufferAttribute(uv, 2));
  g.setIndex(idx);
  g.computeVertexNormals();
  return g;
}

export function buildFalls() {
  const g = new THREE.Group();
  g.name = "falls";
  const rock = makeToonMaterial(0xffffff, { vertexColors: true, surface: "rock", side: THREE.DoubleSide });
  const cliff = new THREE.Mesh(cliffGeometry(), rock);
  cliff.castShadow = cliff.receiveShadow = true;
  g.add(cliff);

  // Pool: a stone-rimmed slab (rim top 1.2 ft) with the water plane just above it, inset so the rim shows.
  const slab = new THREE.Mesh(new THREE.ExtrudeGeometry(shapeOf(poolPolygon(0, 0)), { depth: 1.2, bevelEnabled: false }).rotateX(-Math.PI / 2), makeToonMaterial(0xa49482, { surface: "rock" }));
  slab.receiveShadow = true;
  g.add(slab);
  const waterMat = makeToonMaterial(PALETTE.water, { surface: "none", polygonOffset: true, polygonOffsetFactor: -2, polygonOffsetUnits: -2 });
  const water = new THREE.Mesh(new THREE.ShapeGeometry(shapeOf(poolPolygon(1.0, 3.5))).rotateX(-Math.PI / 2).translate(0, 1.3, 0), waterMat);
  water.receiveShadow = true;
  g.add(water);

  // The fall (animated stripes), running down the face and along the plateau. MeshBasic: unlit, dimmed at night in tick.
  const tex = fallTexture();
  tex.repeat.set(1, 1);
  const fallMat = new THREE.MeshBasicMaterial({ map: tex, transparent: true, opacity: 0.93, side: THREE.DoubleSide });
  const strip = noOutline(new THREE.Mesh(fallStrip(15, [0, 4, 10, 17, 25, 34, 40, FALLS.faceRun - 2, FALLS.faceRun + 2, 70, 100, 130]), fallMat));
  g.add(strip);

  // Where it lands: foam rings on the pool and mist puffs (one instanced mesh, driven in tick).
  const aF = FALLS.aFall;
  const landing = polar(aF, fenceAt(aF) + FALLS.cliffOff - 7);
  const foamMat = new THREE.MeshBasicMaterial({ color: 0xf2fbfa, transparent: true, opacity: 0.7, depthWrite: false, side: THREE.DoubleSide });
  const rings = [0, 1].map(() => {
    const m = noOutline(new THREE.Mesh(new THREE.RingGeometry(0.62, 1, 20).rotateX(-Math.PI / 2), foamMat.clone()));
    m.position.set(landing[0], 1.5, -landing[1]);
    g.add(m);
    return m;
  });
  const PUFFS = 9;
  const mistMat = new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.5, depthWrite: false });
  const mist = noOutline(new THREE.InstancedMesh(new THREE.IcosahedronGeometry(1, 1), mistMat, PUFFS));
  mist.frustumCulled = false;
  g.add(mist);
  const puff = Array.from({ length: PUFFS }, (_, i) => ({ dx: (R() - 0.5) * 26, dr: (R() - 0.5) * 8 - 1, s: 4.5 + R() * 3.5, ph: R() * 6.3, up: 3 + R() * 7 + i * 0.2 }));

  // Boulders round the pool's rim and the cliff's foot (instanced).
  const sites = [];
  for (let k = 0; k < 22; k++) {
    const a = FALLS.a0 + 0.8 + R() * (FALLS.a1 - FALLS.a0 - 1.6);
    const edge = R() < 0.5 ? FALLS.poolIn : FALLS.poolOut;
    sites.push({ p: polar(a, fenceAt(a) + edge + (R() - 0.5) * 3), s: 2 + R() * 4, yaw: R() * 6 });
  }
  const boulders = new THREE.InstancedMesh(new THREE.DodecahedronGeometry(1, 0).scale(1.2, 0.75, 1).translate(0, 0.45, 0), makeToonMaterial(0xffffff, { surface: "rock" }), sites.length);
  const col = new THREE.Color();
  sites.forEach((s, i) => { place(boulders, i, toScene(s.p, 0), s.yaw, s.s); boulders.setColorAt(i, col.set(i % 2 ? PALETTE.rock : 0xa49482)); });
  boulders.castShadow = boulders.receiveShadow = true;
  g.add(boulders);

  // Animation lives on its own empty: a tick on the group would stop mergeStatic merging the cliff and pool.
  const M = new THREE.Matrix4(), Q = new THREE.Quaternion(), S = new THREE.Vector3(), Pv = new THREE.Vector3();
  const tickObj = new THREE.Object3D();
  g.add(tickObj);
  let lastTod = null;
  tickObj.userData.tick = (t) => {
    tex.offset.y = t * 0.9; // downward along the strip's v
    const tod = timeOfDay();
    if (tod !== lastTod) {
      lastTod = tod;
      const k = tod === "night" ? 0.38 : tod === "sunset" ? 0.75 : 1;
      fallMat.color.setScalar(k); foamMat.color.setScalar(k); mistMat.color.setScalar(k);
      for (const r of rings) r.material.color.setScalar(k);
    }
    rings.forEach((r, i) => {
      const u = (t * 0.45 + i * 0.5) % 1;
      r.scale.setScalar(5 + u * 22);
      r.material.opacity = 0.7 * (1 - u);
    });
    puff.forEach((p, i) => {
      const w = 0.5 + 0.5 * Math.sin(t * 0.8 + p.ph);
      const xy = polar(aF, fenceAt(aF) + FALLS.cliffOff - 5 + p.dr);
      Pv.set(xy[0] + p.dx, 1.5 + p.up * (0.7 + 0.5 * w), -xy[1]);
      S.setScalar(p.s * (0.85 + 0.3 * w));
      mist.setMatrixAt(i, M.compose(Pv, Q, S));
    });
    mist.instanceMatrix.needsUpdate = true;
  };
  return g;
}
