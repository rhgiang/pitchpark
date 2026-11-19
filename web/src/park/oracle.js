import * as THREE from "three";
import { mergeGeometries } from "three/examples/jsm/utils/BufferGeometryUtils.js";
import { PALETTE, makeToonMaterial, lampMaterial, noOutline } from "../look.js";
import { place } from "./build.js";
import { palmSites } from "./signature.js";
import { buildWall } from "./wall.js";

/*
 * Oracle-Park-style pieces, in architecture only (no real brands, signs or landmarks):
 *  - a tall brick wall behind the right-field bullpens, an arcade of arched openings you see the Falls
 *    through (and over the top of), with a stone plinth and cornice, a lantern on every pier and
 *    pennant poles along the top;
 *  - palm trees beyond left field.
 * Everything repeated is one InstancedMesh (bays, trim, lanterns, palm trunks, palm fronds), so the
 * whole thing costs about eight draw calls. Statcast ft in, scene ft out (x, y up, z = -Statcast y).
 */

const at = ([x, y], h = 0) => [x, h, -y];

// ---------- palms ----------

const PALM_BEND = 0.14;

/** Unit-height palm trunk: a bent, bumpy five-sided tube, wide at the foot, curving toward +x. */
function trunkGeometry() {
  const RINGS = 9, SIDES = 5;
  const pos = [], idx = [];
  for (let i = 0; i <= RINGS; i++) {
    const t = i / RINGS;
    const cx = PALM_BEND * t * t, y = t;
    const r = (0.05 - 0.028 * t) * (1 + 0.14 * (i % 2));
    for (let k = 0; k < SIDES; k++) {
      const a = (k / SIDES) * Math.PI * 2;
      pos.push(cx + Math.cos(a) * r, y, Math.sin(a) * r);
    }
  }
  for (let i = 0; i < RINGS; i++) for (let k = 0; k < SIDES; k++) {
    const a = i * SIDES + k, b = i * SIDES + ((k + 1) % SIDES);
    idx.push(a, a + SIDES, b, b, a + SIDES, b + SIDES);
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
  geo.setIndex(idx);
  const flat = geo.toNonIndexed();
  flat.computeVertexNormals();
  return flat;
}

/** Unit-height crown: ten drooping, creased fronds fanned round the trunk top. */
function frondsGeometry() {
  const top = [PALM_BEND, 1, 0];
  const pos = [];
  const FR = 10, SEG = 5, LEN = 0.34;
  for (let f = 0; f < FR; f++) {
    const a = (f / FR) * Math.PI * 2 + (f % 2) * 0.15;
    const len = LEN * (0.85 + 0.25 * ((f * 5) % 3) / 2);
    const d = [Math.cos(a), Math.sin(a)];
    const side = [-d[1], d[0]];
    const lift = f % 2 ? 0.05 : 0.11; // alternate fronds sit higher, so the crown has depth
    const ring = [];
    for (let i = 0; i <= SEG; i++) {
      const s = i / SEG;
      const w = 0.045 * Math.sin(Math.PI * Math.min(1, s * 1.1 + 0.08)) ** 0.7;
      const c = [top[0] + d[0] * len * s, top[1] + lift * len * 4 * s * (1 - s) * 2 - 0.14 * len * 4 * s * s, top[2] + d[1] * len * s];
      ring.push([[c[0] - side[0] * w, c[1] - w * 0.5, c[2] - side[1] * w], c, [c[0] + side[0] * w, c[1] - w * 0.5, c[2] + side[1] * w]]);
    }
    for (let i = 0; i < SEG; i++) for (const j of [0, 1]) {
      const [a0, a1, b0, b1] = [ring[i][j], ring[i][j + 1], ring[i + 1][j], ring[i + 1][j + 1]];
      pos.push(...a0, ...b0, ...a1, ...a1, ...b0, ...b1);
    }
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
  geo.computeVertexNormals();
  return geo;
}

function palms() {
  const g = new THREE.Group();
  const sites = palmSites();
  const trunks = new THREE.InstancedMesh(trunkGeometry(), makeToonMaterial(0x8a6a48, { surface: "wood" }), sites.length);
  const crowns = new THREE.InstancedMesh(frondsGeometry(), makeToonMaterial(0xffffff, { surface: "foliage", side: THREE.DoubleSide }), sites.length);
  const c = new THREE.Color();
  const greens = [0x3f8f4a, 0x5ea24c, 0x2f7a45, 0x7fb24a];
  sites.forEach((s, i) => {
    place(trunks, i, at(s.p), s.yaw, s.h);
    place(crowns, i, at(s.p), s.yaw, s.h);
    crowns.setColorAt(i, c.set(greens[i % greens.length]));
  });
  for (const m of [trunks, crowns]) { m.castShadow = true; m.receiveShadow = true; g.add(m); }
  return g;
}

export function buildOracle(flags, poles) {
  const g = new THREE.Group();
  g.name = "oracle";
  g.add(buildWall(flags, poles), palms());
  return g;
}
