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

/**
 * What the scoreboard shows, from the viewer's state (main.js passes it each frame): the count,
 * outs and inning before the current pitch, the pitcher, pitch type and speed (all known at
 * release, as in the scorebug). Runs and hits are left blank: the export has no score, and the
 * board must not invent one.
 * @param info {{balls, strikes, outs, inning, top, pitcher, pitch, mph}} or null (no data yet)
 * @returns {string[]} text lines, also used as the "has it changed" key
 */
export function boardLines(info) {
  if (!info) return ["", "", ""];
  return [
    `${info.top ? "TOP" : "BOT"} ${info.inning}`,
    `${info.balls}-${info.strikes}-${info.outs}`,
    `${(info.pitcher ?? "").toUpperCase()} · ${(info.pitch ?? "").toUpperCase()}${info.mph == null ? "" : ` · ${info.mph.toFixed(1)} MPH`}`,
  ];
}

/** Paint the scoreboard face: title, innings with the current one lit, B/S/O lamps, pitch line. */
function drawBoard(x, info) {
  x.fillStyle = "#1d3550"; x.fillRect(0, 0, 1024, 500);
  x.fillStyle = "#f2c14e"; x.font = "bold 64px sans-serif"; x.textAlign = "center";
  x.fillText("PITCHSIM PARK", 512, 82);
  x.font = "bold 34px sans-serif";
  ["1", "2", "3", "4", "5", "6", "7", "8", "9", "R", "H", "E"].forEach((t, i) => {
    x.fillStyle = info && i === info.inning - 1 ? "#f2c14e" : "#f4ead2";
    x.fillText(t, 250 + i * 62, 160);
  });
  for (let r = 0; r < 2; r++) for (let i = 0; i < 12; i++) {
    // The current half-inning's box glows; the rest stay blank (no score in the data).
    const on = info && i === info.inning - 1 && r === (info.top ? 0 : 1);
    x.fillStyle = on ? "#3fa7a0" : "#2a4a6b";
    x.fillRect(226 + i * 62, 185 + r * 70, 48, 54);
  }
  x.fillStyle = "#f4ead2"; x.textAlign = "left";
  x.fillText("AWAY", 40, 225); x.fillText("HOME", 40, 295);
  const lit = info ? [info.balls, info.strikes, info.outs] : [0, 0, 0];
  const on = ["#3fbf7f", "#e5604b", "#f2c14e"];
  ["BALL", "STRIKE", "OUT"].forEach((t, i) => {
    x.fillStyle = "#f4ead2";
    x.fillText(t, 40 + i * 330, 380);
    for (let k = 0; k < [3, 2, 2][i]; k++) { // lamps: 3 balls, 2 strikes, 2 outs (as on real boards)
      x.beginPath(); x.arc(185 + i * 330 + k * 36, 368, 13, 0, 7);
      x.fillStyle = k < lit[i] ? on[i] : "#2a4a6b"; x.fill();
    }
  });
  x.font = "bold 36px sans-serif"; x.fillStyle = "#f2c14e"; x.textAlign = "center";
  x.fillText(boardLines(info)[2], 512, 460);
}

/** Scoreboard on legs above the right-centre bleachers, showing the live count (update()). */
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
  drawBoard(x, null);
  const tex = Object.assign(new THREE.CanvasTexture(c), { colorSpace: THREE.SRGBColorSpace, anisotropy: 8 });
  // Emissive map: the board is a lit screen, readable at night too (and its bright bits bloom).
  const face = new THREE.Mesh(new THREE.PlaneGeometry(74, 34),
    makeToonMaterial(0xffffff, { map: tex, emissiveMap: tex, emissive: 0xffffff, emissiveIntensity: 0.35, surface: "none" }));
  face.position.set(0, 65, -2.05);
  face.rotation.y = Math.PI; // face home (local -z)
  f.add(face);
  let key = null;
  f.userData.setBoard = (info) => {
    const k = JSON.stringify(info);
    if (k === key) return; // redraw the canvas only when something changed
    key = k;
    drawBoard(x, info);
    tex.needsUpdate = true;
  };
  return f;
}

/**
 * Outfield wall advertising, in the park's own words (no real brands): panels along the fence
 * away from the distance numbers and the centre-field batter's eye (kept dark for hitters).
 */
function wallAds() {
  const g = new THREE.Group();
  const words = ["PITCHSIM", "RK4 PHYSICS", "SEAM WAKE", "TUNNEL VISION", "HELD-OUT", "SPIN AXIS", "STATCAST FIT", "M1 · M2 · M3"];
  const cols = [["#e5604b", "#f4ead2"], ["#f2c14e", "#2a4a6b"], ["#f4ead2", "#2a4a6b"], ["#3fa7a0", "#f4ead2"]];
  let k = 0;
  for (const side of [-1, 1]) for (let a = 13; a <= 41; a += 4.6) {
    if (Math.abs(a - 22.5) < 3) continue; // the gap numbers
    const ang = side * a;
    const a0 = polar(ang - 0.5, fenceDistFt(ang - 0.5)), a1 = polar(ang + 0.5, fenceDistFt(ang + 0.5));
    const f = frameAt(polar(ang, fenceDistFt(ang) - 0.3), [a1[0] - a0[0], a1[1] - a0[1]]);
    const [bg, fg] = cols[k % cols.length];
    const m = new THREE.Mesh(new THREE.PlaneGeometry(22, 4.6),
      // Polygon offset: 0.3 ft off a wall 400 ft away is below the depth buffer's resolution.
      makeToonMaterial(0xffffff, { surface: "none", polygonOffset: true, polygonOffsetFactor: -2, polygonOffsetUnits: -8,
        map: textTexture(words[k % words.length], { w: 640, h: 134, bg, fg, font: "bold 64px sans-serif" }) }));
    m.position.set(0, 3.3, 0);
    f.add(m);
    g.add(f);
    k++;
  }
  return g;
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
  // A shelter with a bench against each pen's back (bleacher) wall: navy roof on steel posts.
  for (const s of [-1, 1]) {
    const q0 = polar(s * 29, fenceDistFt(29) + BLEACHER_GAP_FT - 4), q1 = polar(s * 33, fenceDistFt(33) + BLEACHER_GAP_FT - 4);
    const f = frameAt([(q0[0] + q1[0]) / 2, (q0[1] + q1[1]) / 2], [q1[0] - q0[0], q1[1] - q0[1]]);
    const steel = makeToonMaterial(PALETTE.steel);
    for (const x of [-11, 11]) for (const z of [-2.5, 2.5]) f.add(box(steel, 0.4, 8, 0.4, x, 0, z));
    f.add(box(makeToonMaterial(PALETTE.navy), 25, 0.6, 7, 0, 8, 0));
    f.add(box(makeToonMaterial(PALETTE.coral), 25.2, 0.8, 0.3, 0, 7.6, s > 0 ? -3.6 : 3.6));
    f.add(box(makeToonMaterial(0xb07a4a, { surface: "wood" }), 20, 1.6, 1.6, 0, 0, 0));
    g.add(f);
  }
  return g;
}

/** Instanced trees: conifers (stacked cones) and round trees, with trunks. */
function trees(list) {
  const cone = (r, h, y) => new THREE.ConeGeometry(r, h, 7).translate(0, y + h / 2, 0);
  const conifer = mergeGeometries([cone(0.5, 0.5, 0.18), cone(0.38, 0.42, 0.45), cone(0.25, 0.34, 0.7)]);
  const round = new THREE.IcosahedronGeometry(0.36, 1).scale(1, 0.9, 1).translate(0, 0.62, 0);
  const trunk = new THREE.CylinderGeometry(0.05, 0.07, 0.4, 6).translate(0, 0.2, 0);
  // Chunky "cluster" canopies (three blobs, the cricket clip's voxel-ish trees) and tall poplars,
  // so the island is not one repeated tree.
  const blob = (r, x, y, z) => new THREE.IcosahedronGeometry(r, 0).translate(x, y, z);
  const cluster = mergeGeometries([blob(0.3, 0, 0.62, 0), blob(0.24, 0.2, 0.5, 0.1), blob(0.22, -0.18, 0.52, -0.08), blob(0.2, 0.02, 0.8, 0.05)]);
  const poplar = new THREE.ConeGeometry(0.16, 0.95, 6).scale(1, 1, 1).translate(0, 0.3 + 0.47, 0);
  const kinds = Object.fromEntries(["conifer", "round", "cluster", "poplar"].map((k) => [k, list.filter((t) => t.kind === k)]));
  const out = [];
  const c = new THREE.Color();
  for (const [k, geo] of [["conifer", conifer], ["round", round], ["cluster", cluster], ["poplar", poplar]]) {
    const inst = new THREE.InstancedMesh(geo, makeToonMaterial(0xffffff, { surface: "foliage" }), kinds[k].length);
    kinds[k].forEach((t, i) => {
      place(inst, i, [t.p[0], t.y ?? 0, -t.p[1]], t.yaw, t.h);
      inst.setColorAt(i, c.set(t.color));
    });
    inst.castShadow = inst.receiveShadow = true;
    out.push(inst);
  }
  const tr = new THREE.InstancedMesh(trunk, makeToonMaterial(PALETTE.trunk, { surface: "wood" }), list.length);
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

/** Pennants along the roof's front edge: their poles and flags join the park's waving flags. */
function roofFlags(path, flags, poles) {
  const n = normals(path);
  const cols = [PALETTE.mustard, PALETTE.teal, PALETTE.coral, PALETTE.cream];
  path.forEach((p, i) => {
    if (i % 8 !== 4) return;
    const q = [p[0] + n[i][0] * 57, p[1] + n[i][1] * 57];
    poles.push({ p: q, y: 92.5, s: [1, 10, 1] });
    flags.push({ base: [q[0], 92.5 + 9.8, -q[1]], yaw: 0.6, len: 5, h: 3, color: cols[(i >> 3) % cols.length], pennant: true });
  });
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
    // 0 at the top, 1 at the tip. Clamped: float error can put the tip ring a hair past 1, and a
    // negative base to a fractional power is NaN (three's "computeBoundingSphere: radius is NaN").
    const t = Math.min(1, -y / depth);
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
export function buildProps(plan, flags, poles) {
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
  const board = scoreboard(24, fenceDistFt(24) + BLEACHER_GAP_FT + plan.bleacherDepth + 20);
  g.userData.setBoard = board.userData.setBoard;
  g.add(board);
  g.add(distanceMarkers(), battersEye(), bullpens(), wallAds());
  g.add(...trees(plan.trees), ...umbrellas(plan.umbrellas), ...plan.kiosks.map(kiosk));
  roofFlags(plan.roofPath, flags, poles);
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


