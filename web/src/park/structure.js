import * as THREE from "three";
import { PALETTE, makeToonMaterial } from "../look.js";
import { box, radialFrame } from "./build.js";
import { fenceDistFt, normals, BLEACHER_GAP_FT } from "./layout.js";

/*
 * Structure: dugouts and bullpen fences (?old=structure restores the plain ones). The upper deck's
 * longer reach and the haze are in stands.js / haze.js.
 */

/**
 * Fill a dugout frame f (local x along the foul line, +z into foul ground toward the stands, floor
 * y = 0): raised deck with two end steps, back wall with a team stripe, padded front parapet with a
 * rail, roof with fascia and posts, benches, bat rack, helmets and water cooler.
 * @param L length (ft) along the line, D depth (ft)
 */
export function fillDugout(f, L, D) {
  const mat = (hex, extra) => makeToonMaterial(hex, extra);
  const dark = mat(PALETTE.batterEye), navy = mat(PALETTE.navy), coral = mat(PALETTE.coral), cream = mat(PALETTE.cream);
  const conc = mat(PALETTE.concrete), steel = mat(PALETTE.steel), wood = mat(0xb07a4a, { surface: "wood" });
  const DECK = 1.2; // the deck is a step up from the field (real dugouts are sunk; this reads the same from afar)
  f.add(box(dark, L, 7, 0.6, 0, 0, D / 2 - 0.3)); // back wall
  f.add(box(coral, L, 0.9, 0.7, 0, 3.4, D / 2 - 0.35)); // team stripe
  f.add(box(conc, L, DECK, D, 0, 0, 0)); // deck
  f.add(box(conc, 0.8, 7.5, D, -L / 2, 0, 0), box(conc, 0.8, 7.5, D, L / 2, 0, 0)); // ends
  // Front parapet 3 ft high with a cream padded cap, a gap in the middle of the field side for the top step.
  const z = -D / 2 + 0.3, gap = 7;
  for (const [x0, x1] of [[-L / 2, -gap / 2], [gap / 2, L / 2]]) {
    f.add(box(navy, x1 - x0, 3 - DECK, 0.6, (x0 + x1) / 2, DECK, z));
    f.add(box(cream, x1 - x0 + 0.2, 0.35, 0.9, (x0 + x1) / 2, 3, z));
  }
  for (let k = 0; k < 3; k++) f.add(box(cream, gap, (k + 1) * 0.4, 0.9, 0, 0, z - 0.45 - (2 - k) * 0.9)); // steps up into the gap
  for (const x of [-gap / 2, gap / 2]) f.add(box(steel, 0.16, 2.4, 0.16, x, 3.3, z - 1.2), box(steel, 0.16, 0.16, 2.4, x, 5.5, z - 0.1)); // handrails
  // Roof: slab over the whole dugout, coral fascia, steel posts along the front.
  f.add(box(navy, L + 1.6, 0.7, D + 1.4, 0, 7.5, -0.5));
  f.add(box(coral, L + 1.6, 0.6, 0.25, 0, 7.55, -D / 2 - 1.3));
  for (let x = -L / 2 + 4; x < L / 2; x += 8) f.add(box(steel, 0.25, 4.5, 0.25, x, 3.3, z - 0.1 - 1.2));
  // Benches (two levels), bat rack with bats, helmets, water cooler.
  f.add(box(wood, L - 4, 0.3, 1.6, 0, DECK + 1.2, D / 2 - 1.5), box(cream, L - 4, 1.2, 0.3, 0, DECK, D / 2 - 1.5));
  f.add(box(wood, L - 10, 0.3, 1.4, 0, DECK + 0.1, D / 2 - 3.3));
  f.add(box(steel, 6, 3, 0.3, L / 2 - 6, DECK, D / 2 - 0.9));
  for (let k = 0; k < 8; k++) f.add(box(wood, 0.28, 3.2, 0.28, L / 2 - 8.4 + k * 0.7, DECK + 0.2, D / 2 - 1.2));
  const helmet = makeToonMaterial(PALETTE.navy);
  for (let k = 0; k < 6; k++) {
    const h = new THREE.Mesh(new THREE.SphereGeometry(0.55, 8, 6, 0, Math.PI * 2, 0, Math.PI / 2), helmet);
    h.position.set(-L / 2 + 8 + k * 1.6, DECK + 3.6, D / 2 - 1.1);
    f.add(h);
  }
  const cooler = new THREE.Mesh(new THREE.CylinderGeometry(0.8, 0.8, 1.8, 10), mat(0xe58a3a));
  cooler.position.set(-L / 2 + 3, DECK + 0.9, D / 2 - 2);
  f.add(cooler);
}

/** Bullpen end fences: green chain-link-height walls across the gap behind the outfield fence. */
export function bullpenFences() {
  const g = new THREE.Group();
  const wall = makeToonMaterial(PALETTE.wall), cap = makeToonMaterial(PALETTE.wallTop), steel = makeToonMaterial(PALETTE.steel);
  for (const s of [-1, 1]) for (const a of [25, 39]) {
    const fr = radialFrame(s * a, fenceDistFt(a) + BLEACHER_GAP_FT / 2);
    const len = BLEACHER_GAP_FT - 4;
    fr.add(box(wall, 0.4, 5, len, 0, 0, 0), box(cap, 0.6, 0.3, len, 0, 5, 0));
    for (let k = -len / 2; k <= len / 2 + 1e-6; k += len / 4) fr.add(box(steel, 0.3, 7, 0.3, 0, 0, k));
    g.add(fr);
  }
  return g;
}

/**
 * Open steel truss under the upper deck's roof: every 16 ft a triangle of chords and diagonals
 * (roof underside to the back wall), so the roof reads as steel framing, not a slab.
 * @param path the upper deck path (stands.js upperPath), normals point out
 */
export function roofTruss(path) {
  const g = new THREE.Group();
  const steel = makeToonMaterial(0x2c4a40);
  const n = normals(path);
  // Roof underside runs from (s 56, y 91.6) to (s 99, y 84.6); the truss hangs 7 ft below it at the back.
  const pts = { a: [56, 91.4], b: [99, 84.4], c: [99, 77], d: [78, 88] };
  const member = (f, [z0, y0], [z1, y1], t = 0.5) => {
    const len = Math.hypot(z1 - z0, y1 - y0);
    const m = new THREE.Mesh(new THREE.BoxGeometry(t, t, len), steel);
    m.castShadow = true;
    m.position.set(0, (y0 + y1) / 2, (z0 + z1) / 2);
    m.rotation.x = -Math.atan2(y1 - y0, z1 - z0);
    f.add(m);
  };
  for (let i = 2; i < path.length - 2; i += 4) {
    const f = new THREE.Group();
    f.position.set(path[i][0], 0, -path[i][1]);
    f.rotation.y = Math.atan2(n[i][0], -n[i][1]); // local +z outward
    member(f, pts.a, pts.c); member(f, pts.c, pts.b); member(f, pts.a, pts.d); member(f, pts.d, pts.c);
    g.add(f);
  }
  return g;
}
