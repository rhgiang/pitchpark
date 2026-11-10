import * as THREE from "three";
import { mergeGeometries } from "three/examples/jsm/utils/BufferGeometryUtils.js";
import { PALETTE, makeToonMaterial, lampMaterial, makeGlowSprite, noOutline } from "../look.js";
import { sweep, frameAt, radialFrame, box, place, textTexture, shadeOf } from "./build.js";
import {
  POLE_R, POLE_L, ISLAND, polar, fenceDistFt, normals, rng, BLEACHER_GAP_FT,
  BATTERS_EYE_HALF_DEG, FIELD_CENTRE_FT as FIELD_CENTRE,
} from "./layout.js";

// Everything that isn't seating or ground: poles, towers, scoreboard, batter's eye, bullpens,
// trees, umbrellas, kiosks, flags, the island's rock underside and the floating islets.


/** Light tower: a tapered mast and a bank of lamps aimed at the field, with a night glow. */
function lightTower([x, y], heightFt, lampSlots) {
  const g = frameAt([x, y], [1, 0]);
  const steel = makeToonMaterial(PALETTE.steel);
  const mast = new THREE.Mesh(new THREE.CylinderGeometry(1.6, 3.4, heightFt, 4).rotateY(Math.PI / 4).translate(0, heightFt / 2, 0), steel);
  mast.castShadow = true;
  g.add(mast);
  for (let h = 20; h < heightFt - 10; h += 22) g.add(box(steel, 5.2, 0.8, 5.2, 0, h, 0)); // bracing rings
  // Head, turned to face the field centre and tilted down.
  const head = new THREE.Group();
  head.position.set(0, heightFt, 0);
  head.rotation.y = Math.atan2(-(FIELD_CENTRE[0] - x), FIELD_CENTRE[1] - y); // local -z toward the field
  const tilt = new THREE.Group();
  tilt.rotation.x = -0.35; // aim down at the field
  head.add(tilt);
  tilt.add(box(steel, 38, 22, 2, 0, -2, 0.8));
  tilt.add(box(makeToonMaterial(PALETTE.coral), 38, 1.2, 2.4, 0, 20, 0.8));
  for (let r = 0; r < 4; r++) for (let c = 0; c < 6; c++) {
    lampSlots.push({ parent: tilt, p: [-15 + c * 6, 1.5 + r * 5, -0.6] });
  }
  const glow = makeGlowSprite(60);
  glow.position.set(0, 9, -6);
  tilt.add(glow);
  g.add(head);
  return g;
}

/** Scoreboard on legs above the right-centre bleachers. Decorative only: no numbers shown. */
function scoreboard(angleDeg, r) {
  const f = radialFrame(angleDeg, r);
  const steel = makeToonMaterial(PALETTE.steel);
  f.add(box(steel, 3, 48, 3, -24, 0, 0), box(steel, 3, 48, 3, 24, 0, 0));
  f.add(box(makeToonMaterial(PALETTE.navy), 78, 38, 4, 0, 46, 0));
  f.add(box(makeToonMaterial(PALETTE.coral), 80, 3, 5, 0, 84, 0));
  f.add(box(makeToonMaterial(PALETTE.cream), 30, 7, 3, 0, 87, 0));
  const c = document.createElement("canvas");
  c.width = 1024; c.height = 500;
  const x = c.getContext("2d");
  x.fillStyle = "#1d3550"; x.fillRect(0, 0, 1024, 500);
  x.fillStyle = "#f2c14e"; x.font = "bold 64px sans-serif"; x.textAlign = "center";
  x.fillText("PITCHSIM PARK", 512, 82);
  x.font = "bold 34px sans-serif"; x.fillStyle = "#f4ead2";
  ["1", "2", "3", "4", "5", "6", "7", "8", "9", "R", "H", "E"].forEach((t, i) => x.fillText(t, 250 + i * 62, 160));
  x.fillStyle = "#2a4a6b";
  for (let r = 0; r < 2; r++) for (let i = 0; i < 12; i++) x.fillRect(226 + i * 62, 185 + r * 70, 48, 54);
  x.fillStyle = "#f4ead2"; x.textAlign = "left";
  x.fillText("AWAY", 40, 225); x.fillText("HOME", 40, 295);
  ["BALL", "STRIKE", "OUT"].forEach((t, i) => {
    x.fillText(t, 70 + i * 330, 410);
    for (let k = 0; k < 3; k++) { x.beginPath(); x.arc(200 + i * 330 + k * 34, 398, 12, 0, 7); x.fillStyle = "#2a4a6b"; x.fill(); }
    x.fillStyle = "#f4ead2";
  });
  const face = new THREE.Mesh(new THREE.PlaneGeometry(74, 34),
    makeToonMaterial(0xffffff, { map: Object.assign(new THREE.CanvasTexture(c), { colorSpace: THREE.SRGBColorSpace, anisotropy: 8 }) }));
  face.position.set(0, 65, -2.05);
  face.rotation.y = Math.PI; // face home (local -z)
  f.add(face);
  return f;
}

/** Foul pole: tall yellow pole with a screen on the fair side. */
function foulPole([x, y], side) {
  const g = frameAt([x, y], [Math.SQRT1_2 * side, Math.SQRT1_2]);
  const y0 = makeToonMaterial(PALETTE.mustard);
  const pole = new THREE.Mesh(new THREE.CylinderGeometry(0.6, 0.7, 90, 10).translate(0, 45, 0), y0);
  pole.castShadow = true;
  g.add(pole);
  g.add(box(y0, 0.2, 70, 3, 0, 18, side * -1.8));
  return g;
}

/** Distance numbers painted on the fence, from fenceDistFt (the numbers are the real model). */
function distanceMarkers() {
  const g = new THREE.Group();
  for (const a of [-45, -22.5, 0, 22.5, 45]) {
    const d = fenceDistFt(a);
    const m = new THREE.Mesh(new THREE.PlaneGeometry(9, 4.5),
      makeToonMaterial(0xffffff, { map: textTexture(String(Math.round(d)), { fg: "#f4ead2", font: "bold 110px sans-serif" }), transparent: true, depthWrite: false }));
    // Along the wall itself (it is not square to the line from home: it bends in toward the
    // poles), a hair in front of it; frameAt's local +z then faces the field.
    const a0 = polar(a - 0.5, fenceDistFt(a - 0.5)), a1 = polar(a + 0.5, fenceDistFt(a + 0.5));
    const f = frameAt(polar(a, d - 0.08), [a1[0] - a0[0], a1[1] - a0[1]]);
    m.position.set(Math.abs(a) === 45 ? -Math.sign(a) * 6 : 0, 3.8, 0);
    f.add(noOutline(m));
    g.add(f);
  }
  return g;
}

/** Centre-field batter's eye: a dark wall behind the fence so hitters can see the ball. */
function battersEye() {
  const path = [];
  for (let a = -BATTERS_EYE_HALF_DEG - 1; a <= BATTERS_EYE_HALF_DEG + 1; a += 1) path.push(polar(a, fenceDistFt(a) + 26));
  const m = new THREE.Mesh(sweep(path, [
    [[0, 0], [0, 36], PALETTE.batterEye, 0.7, 1],
    [[0, 36], [3, 36], PALETTE.coral],
    [[3, 36], [3, 0], PALETTE.batterEye],
  ]), makeToonMaterial(0xffffff, { vertexColors: true, side: THREE.DoubleSide }));
  m.castShadow = m.receiveShadow = true;
  return m;
}

/** Bullpens: two mounds and plates per side in the gap behind the fence. */
function bullpens() {
  const g = new THREE.Group();
  const dirt = makeToonMaterial(PALETTE.dirtLight);
  const white = makeToonMaterial(PALETTE.chalk);
  for (const s of [-1, 1]) for (const dr of [-6, 6]) {
    const mound = polar(s * 27, fenceDistFt(27) + BLEACHER_GAP_FT / 2 + dr);
    const plate = polar(s * 36, fenceDistFt(36) + BLEACHER_GAP_FT / 2 + dr);
    const m = new THREE.Mesh(new THREE.CylinderGeometry(2.5, 8, 0.83, 24).translate(0, 0.42, 0), dirt);
    m.position.set(mound[0], 0, -mound[1]);
    m.receiveShadow = true;
    g.add(m, box(white, 2, 0.06, 0.5, mound[0], 0.83, -mound[1]), box(white, 1.4, 0.05, 1.4, plate[0], 0.01, -plate[1]));
  }
  return g;
}

/** Instanced trees: conifers (stacked cones) and round trees, with trunks. */
function trees(list) {
  const cone = (r, h, y) => new THREE.ConeGeometry(r, h, 7).translate(0, y + h / 2, 0);
  const conifer = mergeGeometries([cone(0.5, 0.5, 0.18), cone(0.38, 0.42, 0.45), cone(0.25, 0.34, 0.7)]);
  const round = new THREE.IcosahedronGeometry(0.36, 1).scale(1, 0.9, 1).translate(0, 0.62, 0);
  const trunk = new THREE.CylinderGeometry(0.05, 0.07, 0.4, 6).translate(0, 0.2, 0);
  const kinds = { conifer: list.filter((t) => t.kind === "conifer"), round: list.filter((t) => t.kind === "round") };
  const out = [];
  const c = new THREE.Color();
  for (const [k, geo] of [["conifer", conifer], ["round", round]]) {
    const inst = new THREE.InstancedMesh(geo, makeToonMaterial(0xffffff), kinds[k].length);
    kinds[k].forEach((t, i) => {
      place(inst, i, [t.p[0], t.y ?? 0, -t.p[1]], t.yaw, t.h);
      inst.setColorAt(i, c.set(t.color));
    });
    inst.castShadow = inst.receiveShadow = true;
    out.push(inst);
  }
  const tr = new THREE.InstancedMesh(trunk, makeToonMaterial(PALETTE.trunk), list.length);
  list.forEach((t, i) => place(tr, i, [t.p[0], t.y ?? 0, -t.p[1]], 0, t.h));
  tr.castShadow = true;
  out.push(tr);
  return out;
}

/** Striped parasols with poles (instanced), for the plazas. */
function umbrellas(list) {
  const canopy = new THREE.ConeGeometry(5, 2, 8, 1, true).toNonIndexed().translate(0, 8, 0);
  const pos = canopy.attributes.position;
  const col = [];
  for (let i = 0; i < pos.count; i += 3) {
    const cx = (pos.getX(i) + pos.getX(i + 1) + pos.getX(i + 2)) / 3, cz = (pos.getZ(i) + pos.getZ(i + 1) + pos.getZ(i + 2)) / 3;
    const k = Math.floor(((Math.atan2(cz, cx) + Math.PI) / (2 * Math.PI)) * 8) % 2;
    for (let j = 0; j < 3; j++) col.push(...shadeOf(k ? PALETTE.cream : PALETTE.coral));
  }
  canopy.setAttribute("color", new THREE.Float32BufferAttribute(col, 3));
  canopy.computeVertexNormals();
  const top = new THREE.InstancedMesh(canopy, makeToonMaterial(0xffffff, { vertexColors: true, side: THREE.DoubleSide }), list.length);
  const pole = new THREE.InstancedMesh(new THREE.CylinderGeometry(0.15, 0.15, 8, 6).translate(0, 4, 0), makeToonMaterial(PALETTE.cream), list.length);
  const table = new THREE.InstancedMesh(new THREE.CylinderGeometry(2.2, 2.2, 0.3, 12).translate(0, 2.8, 0), makeToonMaterial(PALETTE.paving), list.length);
  list.forEach((u, i) => {
    for (const m of [top, pole, table]) place(m, i, [u.p[0], 0, -u.p[1]], u.yaw);
  });
  top.castShadow = true;
  return [top, pole, table];
}

/** Food kiosks with striped awnings. */
function kiosk({ p, yaw }) {
  const g = frameAt(p, [Math.cos(yaw), Math.sin(yaw)]);
  g.add(box(makeToonMaterial(PALETTE.cream), 14, 9, 10));
  g.add(box(makeToonMaterial(PALETTE.teal), 15, 1.2, 11, 0, 9, 0));
  for (let i = 0; i < 7; i++) {
    const s = box(makeToonMaterial(i % 2 ? PALETTE.cream : PALETTE.coral), 2, 0.4, 4, -6 + i * 2, 7.4, -6.6);
    s.rotation.x = -0.35;
    g.add(s);
  }
  g.add(box(makeToonMaterial(PALETTE.navy), 10, 3, 0.2, 0, 3, -5.05)); // counter hatch
  return g;
}

/** Pennants along the roof's front edge. */
function flags(path) {
  const n = normals(path);
  const pts = path.filter((_, i) => i % 8 === 4).map((p, k) => ({ p, n: n[path.indexOf(p)], k }));
  const poleG = new THREE.CylinderGeometry(0.15, 0.15, 10, 5).translate(0, 5, 0);
  const flagG = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(0, 10, 0), new THREE.Vector3(0, 7, 0), new THREE.Vector3(5, 8.5, 0)]);
  flagG.computeVertexNormals();
  const poles = new THREE.InstancedMesh(poleG, makeToonMaterial(PALETTE.steelLight), pts.length);
  const fl = new THREE.InstancedMesh(flagG, makeToonMaterial(0xffffff, { side: THREE.DoubleSide }), pts.length);
  const c = new THREE.Color();
  const cols = [PALETTE.mustard, PALETTE.teal, PALETTE.coral, PALETTE.cream];
  pts.forEach(({ p, n: nn, k }, i) => {
    const q = [p[0] + nn[0] * 57, p[1] + nn[1] * 57];
    place(poles, i, [q[0], 92.5, -q[1]]);
    place(fl, i, [q[0], 92.5, -q[1]], 0.6);
    fl.setColorAt(i, c.set(cols[k % cols.length]));
  });
  return [poles, fl];
}

/**
 * The island's underside: a near-vertical cliff of soil (the top 15%), then a jagged cone of
 * rock tapering to a point, flat-shaded (each facet its own normal) and banded with vertex
 * colours, like the cricket clip's island.
 */
function islandRock(centre, radius, depth, seed) {
  const R = rng(seed);
  const g = new THREE.CylinderGeometry(radius, radius, depth, 40, 9, true).translate(0, -depth / 2, 0);
  const pos = g.attributes.position;
  const jit = new Map();
  for (let i = 0; i < pos.count; i++) {
    const x = pos.getX(i), y = pos.getY(i), z = pos.getZ(i);
    const key = `${x.toFixed(2)},${y.toFixed(2)},${z.toFixed(2)}`; // the seam vertices share a jitter
    if (!jit.has(key)) jit.set(key, [R(), R()]);
    const [a, b] = jit.get(key);
    const top = y > -0.01;
    const t = -y / depth; // 0 at the top, 1 at the tip
    const profile = t < 0.15 ? 1 - 0.15 * t : 0.98 * (1 - (t - 0.15) / 0.85) ** 1.4 + 0.03;
    const k = top ? 1 : profile * (1 + (a - 0.5) * 0.22);
    pos.setXYZ(i, x * k, top ? -0.3 : y + (b - 0.5) * depth * 0.05, z * k);
  }
  const geo = g.toNonIndexed();
  geo.computeVertexNormals();
  const p = geo.attributes.position;
  const col = [];
  for (let i = 0; i < p.count; i += 3) {
    const y = (p.getY(i) + p.getY(i + 1) + p.getY(i + 2)) / 3;
    const band = y > -depth * 0.16 ? PALETTE.soil : (Math.floor(-y / (depth * 0.13)) % 2 ? PALETTE.rock : PALETTE.rockDark);
    const s = 0.9 + 0.2 * R();
    for (let j = 0; j < 3; j++) col.push(...shadeOf(band, s));
  }
  geo.setAttribute("color", new THREE.Float32BufferAttribute(col, 3));
  const m = new THREE.Mesh(geo, makeToonMaterial(0xffffff, { vertexColors: true }));
  m.position.set(centre[0], 0, -centre[1]);
  m.castShadow = true;
  return m;
}

/** Small floating islets around the main island: rock, a grass top, and room for trees. */
export const ISLETS = [
  { p: [-760, 620], y: 60, r: 46 }, { p: [820, 380], y: -40, r: 60 }, { p: [700, -330], y: 110, r: 34 },
  { p: [-820, -120], y: -90, r: 52 }, { p: [-280, 1020], y: 20, r: 38 }, { p: [380, 960], y: 150, r: 28 },
];
function islets() {
  const g = new THREE.Group();
  ISLETS.forEach((s, i) => {
    const rock = islandRock([0, 0], s.r, s.r * 1.6, 100 + i);
    const top = new THREE.Mesh(new THREE.CircleGeometry(s.r, 20).rotateX(-Math.PI / 2), makeToonMaterial(PALETTE.grassDeep));
    top.receiveShadow = true;
    const holder = new THREE.Group();
    holder.add(rock, top);
    holder.position.set(s.p[0], s.y, -s.p[1]);
    g.add(holder);
  });
  return g;
}

/**
 * Props, given the plan (tree / umbrella / kiosk placements) and the light towers' sites.
 * @returns {THREE.Group}
 */
export function buildProps(plan) {
  const g = new THREE.Group();
  const lampSlots = [];
  for (const t of plan.towers) g.add(lightTower(t.p, t.h, lampSlots));
  // All lamps as one instanced mesh (world matrices from each tower's tilted head).
  g.updateMatrixWorld(true);
  const lamps = new THREE.InstancedMesh(new THREE.BoxGeometry(4.6, 3.8, 0.8), lampMaterial, lampSlots.length);
  const m = new THREE.Matrix4();
  lampSlots.forEach((s, i) => {
    s.parent.updateMatrixWorld(true);
    lamps.setMatrixAt(i, m.copy(s.parent.matrixWorld).multiply(new THREE.Matrix4().makeTranslation(...s.p)));
  });
  g.add(lamps);
  g.add(foulPole(POLE_R, 1), foulPole(POLE_L, -1));
  g.add(scoreboard(24, fenceDistFt(24) + BLEACHER_GAP_FT + plan.bleacherDepth + 20));
  g.add(distanceMarkers(), battersEye(), bullpens());
  g.add(...trees(plan.trees), ...umbrellas(plan.umbrellas), ...plan.kiosks.map(kiosk), ...flags(plan.roofPath));
  g.add(islandRock(ISLAND.centre, ISLAND.radiusFt, 460, 3), islets());
  for (const b of plan.banners) {
    const f = frameAt(b.p, b.dir);
    const s = new THREE.Mesh(new THREE.PlaneGeometry(24, 4.2),
      makeToonMaterial(0xffffff, { map: textTexture(b.text, { w: 512, h: 90, bg: "#f4ead2", fg: "#2a4a6b", font: "bold 62px sans-serif" }) }));
    s.position.set(0, 38.5, 0.15);
    f.add(s);
    g.add(f);
  }
  return g;
}


