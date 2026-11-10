import * as THREE from "three";
import { PALETTE, makeToonMaterial, noOutline } from "../look.js";
import { sweep, cap, frameAt, box, place } from "./build.js";
import {
  foulWallPath, fencePath, resample, normals, polar, fenceDistFt, rng, BACKSTOP_FT,
  BLEACHER_GAP_FT, BATTERS_EYE_HALF_DEG, FENCE_HEIGHT_FT,
} from "./layout.js";

// Stands: profiles swept along the plan paths (layout.js), then seats and a crowd as instances.
// Profile units: s = ft outward from the wall line, h = ft above the field.

/** Stepped seat rows from [s, h]: returns profile segments and the rows' tread fronts. */
function rows(s, h, n, rise, tread, hex = PALETTE.concrete) {
  const segs = [], treads = [];
  for (let r = 0; r < n; r++) {
    segs.push([[s, h], [s, h + rise], PALETTE.concreteShade, 0.85, 1]); // riser
    h += rise;
    segs.push([[s, h], [s + tread, h], hex, 1, 0.72]); // tread, darker at the back
    treads.push([s, h]);
    s += tread;
  }
  return { segs, treads, end: [s, h] };
}
const outline = (segs) => segs.map((g) => g[0]).concat([segs[segs.length - 1][1]]);

// Lower bowl: 4.5 ft padded wall, 18 rows (1.3 ft rise, 2.9 ft tread), concourse, back wall.
const LOWER_ROWS = rows(1.5, 4.5, 18, 1.3, 2.9);
const LOWER = [
  [[0, 0], [0, 4.5], PALETTE.navy, 0.75, 1],
  [[0, 4.5], [1.5, 4.5], PALETTE.cream],
  ...LOWER_ROWS.segs,
  [LOWER_ROWS.end, [62, LOWER_ROWS.end[1]], PALETTE.paving, 1, 0.8],
  [[62, LOWER_ROWS.end[1]], [62, 34], PALETTE.cream, 0.8, 1],
  [[62, 34], [64, 34], PALETTE.coral],
  [[64, 34], [64, 12], PALETTE.cream, 1, 0.8],
  [[64, 12], [64, 0], PALETTE.teal, 1, 0.7], // plinth
];
// Upper deck (behind home and down to the knees of the foul walls): overhangs the lower
// concourse; 14 steeper rows; back wall up to the roof. Fascia in coral.
const UPPER_ROWS = rows(50, 41, 14, 1.9, 3.0);
const UPPER = [
  [[64, 36], [50, 36], PALETTE.concreteShade, 0.6, 0.8], // underside: in shade
  [[50, 36], [50, 41], PALETTE.coral],
  ...UPPER_ROWS.segs,
  [UPPER_ROWS.end, [96, UPPER_ROWS.end[1]], PALETTE.paving, 1, 0.8],
  [[96, UPPER_ROWS.end[1]], [96, 86], PALETTE.cream, 0.8, 1],
  [[96, 86], [98, 86], PALETTE.coral],
  [[98, 86], [98, 80], PALETTE.coral],
  [[98, 80], [98, 12], PALETTE.cream, 1, 0.8],
  [[98, 12], [98, 0], PALETTE.teal, 1, 0.7], // plinth
];
export const MAIN_DEPTH_FT = 98;
// Roof: a thin slab cantilevered forward from the back wall, striped coral and cream.
const ROOF = [
  [[99, 86], [56, 93], PALETTE.coral],
  [[56, 93], [56, 91.6], PALETTE.coral],
  [[56, 91.6], [99, 84.6], PALETTE.coral, 0.7, 0.7],
];
// Outfield bleachers: a 9 ft wall to see over the fence, 12 rows, railing.
const BLEACH_ROWS = rows(0, 9, 12, 1.5, 3.0);
const BLEACHERS = [
  [[0, 0], [0, 9], PALETTE.wall, 0.75, 1],
  ...BLEACH_ROWS.segs,
  [BLEACH_ROWS.end, [BLEACH_ROWS.end[0] + 2, BLEACH_ROWS.end[1]], PALETTE.paving],
  [[BLEACH_ROWS.end[0] + 2, BLEACH_ROWS.end[1]], [BLEACH_ROWS.end[0] + 2, BLEACH_ROWS.end[1] + 4], PALETTE.coral],
  [[BLEACH_ROWS.end[0] + 2, BLEACH_ROWS.end[1] + 4], [BLEACH_ROWS.end[0] + 3, BLEACH_ROWS.end[1] + 4], PALETTE.cream],
  [[BLEACH_ROWS.end[0] + 3, BLEACH_ROWS.end[1] + 4], [BLEACH_ROWS.end[0] + 3, 0], PALETTE.cream, 1, 0.65],
];
export const BLEACHER_DEPTH_FT = BLEACH_ROWS.end[0] + 3;
// Outfield fence: dark green padded wall with a yellow home-run line along the top.
const FENCE = [
  [[0, 0], [0, FENCE_HEIGHT_FT - 0.4], PALETTE.wall, 0.7, 1],
  [[0, FENCE_HEIGHT_FT - 0.4], [0, FENCE_HEIGHT_FT], PALETTE.wallTop],
  [[0, FENCE_HEIGHT_FT], [1, FENCE_HEIGHT_FT], PALETTE.wallTop],
  [[1, FENCE_HEIGHT_FT], [1, 0], PALETTE.wall],
];

/** The main stands' path: foul corner to foul corner (not the poles: bleachers are there). */
export function mainPath() {
  const p = foulWallPath();
  return resample(p.slice(1, -1), 4);
}
/** The knee-to-knee part of the main path, where the upper deck and roof are. */
export function upperPath() {
  const p = foulWallPath();
  return resample(p.slice(2, -2), 4);
}
/** Bleacher paths: one per side, from the batter's eye to near the pole, BLEACHER_GAP_FT out. */
export function bleacherPaths() {
  const side = (a0, a1) => {
    const pts = [];
    for (let a = a0; a <= a1 + 1e-9; a += 1) pts.push(polar(a, fenceDistFt(a) + BLEACHER_GAP_FT));
    return pts;
  };
  return [side(-42, -BATTERS_EYE_HALF_DEG), side(BATTERS_EYE_HALF_DEG, 42)];
}

/** A material for vertex-coloured sweeps (the colour lives in the vertices). */
const swept = () => makeToonMaterial(0xffffff, { vertexColors: true, side: THREE.DoubleSide });

function mesh(geo, mat, cast = true) {
  const m = new THREE.Mesh(geo, mat);
  m.castShadow = cast;
  m.receiveShadow = true;
  return m;
}

/** Seats on every tread of a swept stand; returns seat positions for the crowd. */
function seatPositions(path, treads, spacing, aisleEvery, out) {
  const n = normals(path);
  for (const [s, h] of treads) {
    const row = resample(path.map((p, i) => [p[0] + n[i][0] * (s + 1.1), p[1] + n[i][1] * (s + 1.1)]), spacing);
    const rn = normals(row);
    row.forEach((p, i) => {
      if (i % aisleEvery >= aisleEvery - 2 || i < 1 || i > row.length - 2) return; // aisles
      out.push({ p, h, yaw: Math.atan2(rn[i][0], -rn[i][1]) }); // seat back (+z) outward: faces the field
    });
  }
  return out;
}

/** Seats and seated crowd, as three InstancedMeshes (seats, bodies, heads). */
function seatsAndCrowd(seats, seed) {
  const R = rng(seed);
  // A seat: pan and back, 1.6 ft wide; back on the far side from the field (+z local is away).
  const seatGeo = new THREE.BoxGeometry(1.6, 0.3, 1.3).translate(0, 1.2, 0);
  const backGeo = new THREE.BoxGeometry(1.6, 1.3, 0.25).translate(0, 1.9, 0.6);
  const seatMesh = new THREE.InstancedMesh(
    mergeTwo(seatGeo, backGeo), makeToonMaterial(0xffffff), seats.length);
  const people = seats.filter(() => R() < 0.5); // half full: the seat colours still show
  const body = new THREE.InstancedMesh(new THREE.BoxGeometry(1.25, 1.7, 0.9).translate(0, 2.15, 0.15),
    makeToonMaterial(0xffffff), people.length);
  const head = new THREE.InstancedMesh(new THREE.IcosahedronGeometry(0.42, 1).translate(0, 3.45, 0.1),
    makeToonMaterial(0xffffff), people.length);
  const c = new THREE.Color();
  seats.forEach((s, i) => {
    place(seatMesh, i, [s.p[0], s.h, -s.p[1]], s.yaw);
    seatMesh.setColorAt(i, c.set(s.color));
  });
  const skins = [0xf1c7a1, 0xd9a07a, 0xb57a52, 0x8a5a3a, 0x5e3b28];
  people.forEach((s, i) => {
    const k = 0.9 + 0.2 * R();
    place(body, i, [s.p[0], s.h, -s.p[1]], s.yaw + (R() - 0.5) * 0.4, [1, k, 1]);
    place(head, i, [s.p[0], s.h + 3.45 * (k - 1), -s.p[1]], s.yaw);
    body.setColorAt(i, c.set(PALETTE.crowd[Math.floor(R() * PALETTE.crowd.length)]));
    head.setColorAt(i, c.set(skins[Math.floor(R() * skins.length)]));
  });
  for (const m of [seatMesh, body, head]) { m.receiveShadow = true; m.castShadow = m === seatMesh; }
  return [seatMesh, body, head];
}

function mergeTwo(a, b) {
  const g = new THREE.BufferGeometry();
  const pos = [...a.attributes.position.array, ...b.attributes.position.array];
  const nor = [...a.attributes.normal.array, ...b.attributes.normal.array];
  const off = a.attributes.position.count;
  g.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
  g.setAttribute("normal", new THREE.Float32BufferAttribute(nor, 3));
  g.setIndex([...a.index.array, ...[...b.index.array].map((i) => i + off)]);
  return g;
}

/** Dugout: open-fronted shelter in foul ground against the stands (local x along the line). */
function dugout(side) {
  const u = [side * Math.SQRT1_2, Math.SQRT1_2]; // along the line
  const n = [side * Math.SQRT1_2, -Math.SQRT1_2]; // into foul ground
  const at = (a, b) => [u[0] * a + n[0] * b, u[1] * a + n[1] * b];
  // Local frame: x along the line (toward home on the 3B side), z into foul ground (toward the stands).
  const f = frameAt(at(50, 55), side > 0 ? u : [-u[0], -u[1]]);
  const L = 48, D = 10;
  const dark = makeToonMaterial(PALETTE.batterEye);
  const roof = makeToonMaterial(PALETTE.navy);
  const trim = makeToonMaterial(PALETTE.coral);
  const conc = makeToonMaterial(PALETTE.concrete);
  f.add(box(dark, L, 7, 0.6, 0, 0, D / 2 - 0.3)); // back wall
  f.add(box(dark, L, 0.05, D, 0, 0.01, 0)); // floor
  f.add(box(conc, 0.8, 7.5, D, -L / 2, 0, 0), box(conc, 0.8, 7.5, D, L / 2, 0, 0)); // ends
  f.add(box(roof, L + 1.6, 0.7, D + 1, 0, 7.5, -0.5)); // roof
  f.add(box(trim, L + 1.6, 0.5, 0.2, 0, 7.6, -D / 2 - 1)); // fascia
  f.add(box(makeToonMaterial(PALETTE.cream), L - 2, 1.4, 1.6, 0, 0, D / 2 - 1.6)); // bench
  f.add(box(makeToonMaterial(PALETTE.steel), L, 0.25, 0.25, 0, 3.4, -D / 2 + 0.3)); // rail
  for (let x = -L / 2 + 4; x < L / 2; x += 8) f.add(box(makeToonMaterial(PALETTE.steel), 0.2, 3.5, 0.2, x, 0, -D / 2 + 0.3));
  return f;
}

/** Backstop net behind home: a fine net on posts, from the wall top to 32 ft. */
function backstop() {
  const g = new THREE.Group();
  const path = [];
  for (let a = 128; a <= 232; a += 2) path.push(polar(a, BACKSTOP_FT + 0.6));
  const c = document.createElement("canvas");
  c.width = c.height = 64;
  const x = c.getContext("2d");
  x.strokeStyle = "rgba(40,50,55,0.9)"; x.lineWidth = 3;
  x.strokeRect(0, 0, 64, 64);
  const tex = new THREE.CanvasTexture(c);
  tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
  tex.repeat.set(140, 14);
  const net = new THREE.Mesh(sweep(path, [[[0, 4.5], [0, 32], 0xffffff]]),
    new THREE.MeshBasicMaterial({ map: tex, transparent: true, side: THREE.DoubleSide, depthWrite: false, opacity: 0.55 }));
  // sweep() has no UVs: give the strip u along the path, v up.
  const uv = [];
  path.forEach((_, i) => uv.push(i / (path.length - 1), 0, i / (path.length - 1), 1));
  net.geometry.setAttribute("uv", new THREE.Float32BufferAttribute(uv, 2));
  g.add(noOutline(net));
  const post = makeToonMaterial(PALETTE.steel);
  for (let i = 0; i < path.length; i += 8) {
    const m = new THREE.Mesh(new THREE.CylinderGeometry(0.25, 0.25, 32, 8), post);
    m.position.set(path[i][0], 16, -path[i][1]);
    g.add(m);
  }
  const top = new THREE.Mesh(sweep(path, [[[-0.2, 32], [0.2, 32], PALETTE.steel], [[0.2, 32], [0.2, 31.6], PALETTE.steel]]), swept());
  g.add(top);
  return g;
}

/**
 * Facade detail on the stands' outer walls: pilasters every 32 ft and dark window slots, as two
 * InstancedMeshes, so the plain back walls read as buildings from the plaza and the wide view.
 * @param walls [{path, s, top, rows}]: wall at offset s (ft) along path, height top, window
 *   rows at heights `rows`.
 */
function facades(walls) {
  const pil = [], win = [];
  for (const { path, s, top, rows } of walls) {
    const n = normals(path);
    const line = resample(path.map((p, i) => [p[0] + n[i][0] * s, p[1] + n[i][1] * s]), 8);
    const ln = normals(line);
    line.forEach((p, i) => {
      if (i === 0 || i === line.length - 1) return;
      const yaw = Math.atan2(ln[i][0], -ln[i][1]);
      if (i % 4 === 0) pil.push({ p, yaw, h: top });
      else for (const h of rows) win.push({ p, yaw, h });
    });
  }
  const pm = new THREE.InstancedMesh(new THREE.BoxGeometry(2.4, 1, 2.2).translate(0, 0.5, 0),
    makeToonMaterial(PALETTE.concrete), pil.length);
  pil.forEach((q, i) => place(pm, i, [q.p[0], 0, -q.p[1]], q.yaw, [1, q.h, 1]));
  const wm = new THREE.InstancedMesh(new THREE.BoxGeometry(4.2, 5, 1.2), makeToonMaterial(PALETTE.navy), win.length);
  win.forEach((q, i) => place(wm, i, [q.p[0], q.h, -q.p[1]], q.yaw));
  pm.castShadow = true;
  pm.receiveShadow = wm.receiveShadow = true;
  return [pm, wm];
}

/** All the seating, walls, dugouts and backstop. */
export function buildStands() {
  const g = new THREE.Group();
  const mat = swept();
  const main = mainPath(), upper = upperPath();
  g.add(mesh(sweep(main, LOWER), mat));
  g.add(mesh(cap(main, 0, outline(LOWER), PALETTE.cream), mat), mesh(cap(main, main.length - 1, outline(LOWER), PALETTE.cream), mat));
  g.add(mesh(sweep(upper, UPPER), mat));
  const upperOutline = [[64, 0], ...outline(UPPER).slice(0, -1)];
  g.add(mesh(cap(upper, 0, upperOutline, PALETTE.cream), mat), mesh(cap(upper, upper.length - 1, upperOutline, PALETTE.cream), mat));
  g.add(mesh(sweep(upper, ROOF, { stripe: 3, stripeHex: PALETTE.cream }), mat));
  // Fence, plus the short corner walls from each pole into foul ground.
  const fence = fencePath(0.5);
  g.add(mesh(sweep(fence, FENCE), mat));
  for (const pts of [[foulWallPath()[0], foulWallPath()[1]], [foulWallPath().at(-2), foulWallPath().at(-1)]]) {
    g.add(mesh(sweep(resample(pts, 2), FENCE), mat));
  }
  for (const b of bleacherPaths()) {
    const p = resample(b, 4);
    g.add(mesh(sweep(p, BLEACHERS), mat));
    const o = outline(BLEACHERS).concat([[BLEACHER_DEPTH_FT, 0]]);
    g.add(mesh(cap(p, 0, o, PALETTE.cream), mat), mesh(cap(p, p.length - 1, o, PALETTE.cream), mat));
  }

  // Seats, coloured by deck so the bowl reads in bands from afar (as the cricket park's do):
  // teal lower bowl, coral upper deck, mustard bleachers.
  const seats = [];
  seatPositions(main, LOWER_ROWS.treads, 2, 16, seats).forEach((s) => (s.color = PALETTE.seat));
  const up0 = seats.length;
  seatPositions(upper, UPPER_ROWS.treads, 2, 16, seats);
  seats.slice(up0).forEach((s) => (s.color = PALETTE.coral));
  for (const b of bleacherPaths()) {
    const k = seats.length;
    seatPositions(resample(b, 4), BLEACH_ROWS.treads, 2, 18, seats);
    seats.slice(k).forEach((s) => (s.color = PALETTE.mustard));
  }
  g.add(...seatsAndCrowd(seats, 11));
  const fw = foulWallPath();
  g.add(...facades([
    { path: upper, s: MAIN_DEPTH_FT, top: 86, rows: [10, 22, 46, 58, 70] },
    { path: resample([fw[1], fw[2]], 4), s: 64, top: 34, rows: [10, 22] },
    { path: resample([fw.at(-3), fw.at(-2)], 4), s: 64, top: 34, rows: [10, 22] },
    ...bleacherPaths().map((b) => ({ path: resample(b, 4), s: BLEACHER_DEPTH_FT, top: BLEACH_ROWS.end[1] + 4, rows: [8, 18] })),
  ]));
  g.add(dugout(1), dugout(-1), backstop());
  g.userData.seatCount = seats.length;
  return g;
}


