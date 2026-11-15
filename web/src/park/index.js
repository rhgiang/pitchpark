import * as THREE from "three";
import { PALETTE, setParkLighting } from "../look.js";
import { buildGround, offsetPath } from "./ground.js";
import { buildStands, mainPath, upperPath, bleacherPaths, MAIN_DEPTH_FT, BLEACHER_DEPTH_FT } from "./stands.js";
import { buildProps, ISLETS } from "./props.js";
import { buildSurrounds, buildFlags } from "./surrounds.js";
import { mergeStatic } from "./build.js";
import { buildBrand, PARK_NAME } from "./brand.js";
import {
  ISLAND, fieldBoundary, fencePath, polar, fenceDistFt, resample, normals, rng, inside,
  BATTERS_EYE_HALF_DEG, FIELD_CENTRE_FT,
} from "./layout.js";

/*
 * The starter ballpark. HOW TO REPLACE IT (Ryan): main.js only calls buildPark() and adds the
 * Group it returns, so a new park only has to return a THREE.Group in scene feet (Statcast
 * (x, y, z) -> scene (x, z, -y): see coords.js), with home plate's back point at the origin and
 * the field at y = 0. The real playing geometry (plate, rubber, mound) is NOT here: it lives in
 * field.js and stays put whatever park you load.
 *
 * - Your own coded park: write park/mypark.js exporting buildPark(), and change the import in
 *   main.js.
 * - A Blender park, later: export glTF (+Y up, 1 unit = 1 ft, or scale on load), then
 *     import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
 *     const gltf = await new GLTFLoader().loadAsync("/park.glb"); return gltf.scene;
 *   (buildPark would become async; main.js awaits it). Swap each mesh's material for
 *   look.js's makeToonMaterial(color) so it gets the outlines' companion shading, fog and the
 *   night falloff, and set castShadow / receiveShadow. Keep the field clear around home plate
 *   and the mound so field.js's geometry shows through.
 * - Lighting that depends on the park's size is the park's job too: call look.js's
 *   setParkLighting({ centreFt, floodRadiiFt, shadowHalfFt }) from buildPark (see below):
 *   the bowl's centre, the radii the night floodlights reach, the half-width of the sun's
 *   shadow box (make it cover your stands; more area means coarser shadows), and where the night
 *   floodlight pools fall (poolsFt).
 * - Optional per-frame hook: if the Group has userData.update({ timeS, board }), main.js calls it
 *   every frame (timeS: seconds since load, for animation; board: the count, outs, inning and
 *   pitch for a scoreboard, or null). A Blender park can leave it out.
 *
 * Layout: everything is placed from a "plan" (paths, footprints, scatter of trees and props)
 * built once from a fixed seed, so the park is the same on every load and the painted ground
 * (contact shadows under trees, plazas round the stands) matches the 3D props exactly.
 */

/** Smallest distance (ft) from p to any point of a densely sampled polyline. */
const distTo = (p, pts) => Math.min(...pts.map((q) => Math.hypot(p[0] - q[0], p[1] - q[1])));

function makePlan() {
  const R = rng(2026);
  const main = mainPath();
  const upper = upperPath();
  const bleachers = bleacherPaths().map((b) => resample(b, 4));
  const fence = resample(fencePath(1), 6);
  const boundary = fieldBoundary();
  const [cx, cy] = ISLAND.centre;
  const plan = {
    mainOuter: offsetPath(main, MAIN_DEPTH_FT),
    bleacherOuter: bleachers.map((b) => offsetPath(b, BLEACHER_DEPTH_FT)),
    bleacherDepth: BLEACHER_DEPTH_FT,
    roofPath: upper,
    trees: [], umbrellas: [], kiosks: [], towers: [], banners: [], paths: [],
    pond: { p: [-430, 330], rx: 62, ry: 40 },
    // Art pass 2 sites (Statcast ft). Gate: behind home, facing out to the path at 180 deg.
    gate: { p: [0, -206], yaw: 0, r: 62 },
    gazebo: [-372, 418],
    lot: { c: [392, -40], w: 120, h: 76 }, // car park, axis-aligned
    fountains: [], tents: [], cars: [], lamps: [], benches: [], bushes: [], rocks: [], hanging: [],
    hangingCentre: ISLAND.centre,
  };
  // The stream runs from the pond to the island's edge, where it falls (the waterfall).
  {
    const d = [plan.pond.p[0] - cx, plan.pond.p[1] - cy];
    const n = Math.hypot(...d);
    const dir = [d[0] / n, d[1] / n];
    plan.waterfall = { p: [cx + dir[0] * (ISLAND.radiusFt - 2), cy + dir[1] * (ISLAND.radiusFt - 2)], dir };
    plan.stream = resample([plan.pond.p, plan.waterfall.p], 6);
  }

  // Light towers: four behind the main stands, two behind the outfield bleachers.
  const back = offsetPath(main, MAIN_DEPTH_FT + 6);
  for (const f of [0.12, 0.33, 0.67, 0.88]) plan.towers.push({ p: back[Math.round(f * (back.length - 1))], h: 150 });
  for (const s of [-1, 1]) plan.towers.push({ p: polar(s * 33, fenceDistFt(33) + 32 + BLEACHER_DEPTH_FT + 8), h: 125 });

  // Paths from the plazas out to the island's edge (gentle curves).
  for (const a of [180, 138, 222, 98, 262, 0, 40, -40]) {
    const t = (a * Math.PI) / 180;
    const s = [cx + Math.sin(t) * 330, cy + Math.cos(t) * 330];
    const e = [cx + Math.sin(t) * (ISLAND.radiusFt + 10), cy + Math.cos(t) * (ISLAND.radiusFt + 10)];
    const m = [(s[0] + e[0]) / 2 + 35 * Math.cos(t), (s[1] + e[1]) / 2 - 35 * Math.sin(t)];
    const pts = [];
    for (let k = 0; k <= 20; k++) {
      const u = k / 20;
      pts.push([0, 1].map((j) => (1 - u) ** 2 * s[j] + 2 * u * (1 - u) * m[j] + u * u * e[j]));
    }
    plan.paths.push(pts);
  }
  const allPath = plan.paths.flat();

  // Plaza furniture: umbrellas and kiosks in the paved ring outside the main stands.
  const plaza = offsetPath(main, MAIN_DEPTH_FT + 22);
  for (let i = 6; i < plaza.length - 6; i += 5) {
    if (R() < 0.45) continue;
    const n = normals(plaza)[i];
    const d = (R() - 0.5) * 16;
    plan.umbrellas.push({ p: [plaza[i][0] + n[0] * d, plaza[i][1] + n[1] * d], yaw: R() * 6 });
  }
  const kioskRing = offsetPath(main, MAIN_DEPTH_FT + 40);
  const kn = normals(kioskRing);
  for (const f of [0.06, 0.2, 0.42, 0.58, 0.8, 0.94]) {
    const i = Math.round(f * (kioskRing.length - 1));
    plan.kiosks.push({ p: kioskRing[i], yaw: Math.atan2(kn[i][1], kn[i][0]) + Math.PI / 2 });
  }

  // Nothing else stands in the gate's footprint.
  const nearGate = (q, r = plan.gate.r) => Math.hypot(q[0] - plan.gate.p[0], q[1] - plan.gate.p[1]) < r;
  plan.umbrellas = plan.umbrellas.filter((u) => !nearGate(u.p));
  // Fountains in the plaza ring, between the kiosks.
  for (const f of [0.31, 0.69]) plan.fountains.push(kioskRing[Math.round(f * (kioskRing.length - 1))]);
  // Lamp posts every ~45 ft along each path (alternating sides), benches between them.
  for (const path of plan.paths) {
    const pts = resample(path, 45), pn = normals(pts);
    pts.forEach((q, i) => {
      if (i === 0) return;
      const side = i % 2 ? 1 : -1;
      plan.lamps.push({ p: [q[0] + pn[i][0] * 11 * side, q[1] + pn[i][1] * 11 * side], yaw: Math.atan2(pn[i][1] * side, pn[i][0] * side) + Math.PI });
      const b = pts[i - 1], m = [(q[0] + b[0]) / 2, (q[1] + b[1]) / 2];
      if (R() < 0.6) plan.benches.push({ p: [m[0] - pn[i][0] * 11 * side, m[1] - pn[i][1] * 11 * side], yaw: Math.atan2(pn[i][1], pn[i][0]) + (side > 0 ? Math.PI / 2 : -Math.PI / 2) });
    });
  }
  // Food tents: two rows outside the right-field bleachers.
  for (let k = 0; k < 8; k++) plan.tents.push({ p: [268 + (k % 4) * 22, 430 + Math.floor(k / 4) * 24], yaw: 0, color: [PALETTE.coral, PALETTE.teal, PALETTE.mustard, PALETTE.coral][k % 4] });
  // Car park: rows of cars, a few gaps.
  const lot = plan.lot;
  for (let r = 0; r < 4; r++) for (let k = 0; k < 7; k++) {
    if (R() < 0.25) continue;
    const carCols = [PALETTE.coral, PALETTE.teal, PALETTE.mustard, PALETTE.cream, PALETTE.navy, 0xd98aa0, 0x6fc3d6];
    plan.cars.push({ p: [lot.c[0] - lot.w / 2 + 12 + k * 16, lot.c[1] - lot.h / 2 + 9 + r * 19 + (r % 2 ? -2 : 2)], yaw: Math.PI / 2, color: carCols[Math.floor(R() * carCols.length)] });
  }
  const inLot = (q, m = 0) => Math.abs(q[0] - lot.c[0]) < lot.w / 2 + m && Math.abs(q[1] - lot.c[1]) < lot.h / 2 + m;

  // Trees: rejection-sampled over the island, clear of the field, stands, plazas, paths, pond.
  const furniture = [...plan.umbrellas.map((u) => u.p), ...plan.kiosks.map((k) => k.p), ...plan.towers.map((t) => t.p),
    ...plan.fountains, ...plan.lamps.map((l) => l.p), ...plan.benches.map((b) => b.p), ...plan.tents.map((t) => t.p), plan.gazebo];
  const bleachAll = bleachers.flat();
  for (let tries = 0; tries < 4000 && plan.trees.length < 760; tries++) {
    const a = R() * 2 * Math.PI, r = Math.sqrt(R()) * (ISLAND.radiusFt - 14);
    const p = [cx + r * Math.cos(a), cy + r * Math.sin(a)];
    if (inside(p, boundary)) continue;
    if (distTo(p, main) < MAIN_DEPTH_FT + 52 || distTo(p, fence) < 26 + 32 + BLEACHER_DEPTH_FT) continue;
    if (distTo(p, bleachAll) < BLEACHER_DEPTH_FT + 30) continue;
    if (distTo(p, allPath) < 13 || distTo(p, furniture) < 14) continue;
    if (Math.hypot((p[0] - plan.pond.p[0]) / (plan.pond.rx + 10), (p[1] - plan.pond.p[1]) / (plan.pond.ry + 10)) < 1) continue;
    if (nearGate(p, plan.gate.r + 20) || inLot(p, 14) || distTo(p, [plan.gazebo]) < 34 || distTo(p, plan.stream) < 16) continue;
    if (Math.hypot(p[0] - 312, p[1] - 442) < 70) continue; // the tents
    plan.trees.push(tree(R, p, 26 + R() * 30));
  }
  // Bushes: hedging along the paths and round the plaza edge, plus a scatter; and boulders.
  for (let tries = 0; tries < 6000 && plan.bushes.length < 420; tries++) {
    const a = R() * 2 * Math.PI, r = Math.sqrt(R()) * (ISLAND.radiusFt - 8);
    const p = [cx + r * Math.cos(a), cy + r * Math.sin(a)];
    if (inside(p, boundary) || distTo(p, main) < MAIN_DEPTH_FT + 38 || distTo(p, bleachAll) < BLEACHER_DEPTH_FT + 18) continue;
    if (distTo(p, fence) < 32 + BLEACHER_DEPTH_FT + 12) continue;
    const dp = distTo(p, allPath);
    if (dp < 10 || (dp > 18 && R() < 0.7)) continue; // mostly lining the paths
    if (distTo(p, furniture) < 6 || nearGate(p) || inLot(p, 4)) continue;
    if (Math.hypot((p[0] - plan.pond.p[0]) / (plan.pond.rx + 4), (p[1] - plan.pond.p[1]) / (plan.pond.ry + 4)) < 1) continue;
    plan.bushes.push({ p, yaw: R() * 6.3, s: 3 + R() * 3, color: PALETTE.foliage[Math.floor(R() * PALETTE.foliage.length)] });
  }
  for (let k = 0; k < 70; k++) {
    const a = R() * 2 * Math.PI, r = (0.55 + 0.45 * Math.sqrt(R())) * (ISLAND.radiusFt - 10);
    const p = [cx + r * Math.cos(a), cy + r * Math.sin(a)];
    if (inside(p, boundary) || distTo(p, allPath) < 10 || inLot(p, 6) || nearGate(p)) continue;
    plan.rocks.push({ p, yaw: R() * 6.3, s: 2 + R() * 5, color: R() < 0.5 ? PALETTE.rock : 0xa49482 });
  }
  // Rocks hanging under the island (offsets from its centre; y below the ground).
  for (let k = 0; k < 16; k++) {
    const a = R() * 2 * Math.PI, r = 120 + R() * 380;
    plan.hanging.push({ p: [Math.cos(a) * r, Math.sin(a) * r], y: -260 - R() * 260 - (380 - r) * 0.4, yaw: R() * 6, s: 6 + R() * 16, color: R() < 0.5 ? PALETTE.rock : PALETTE.rockDark });
  }
  // Batter's eye planting: small conifers in the strip in front of the dark wall.
  for (let a = -BATTERS_EYE_HALF_DEG + 1; a <= BATTERS_EYE_HALF_DEG - 1; a += 2.2) {
    const t = tree(R, polar(a, fenceDistFt(a) + 14), 11 + R() * 4);
    t.kind = "conifer";
    plan.trees.push(t);
  }
  plan.groundTrees = plan.trees.slice();
  // A few trees on each floating islet.
  for (const s of ISLETS) for (let k = 0; k < 3; k++) {
    const t = tree(R, [s.p[0] + (R() - 0.5) * s.r, s.p[1] + (R() - 0.5) * s.r], 16 + R() * 14);
    t.y = s.y;
    plan.trees.push(t);
  }

  // Banners along the upper-deck fascia, behind home.
  const fascia = offsetPath(upper, 49.85);
  const words = [PARK_NAME, "STATCAST FIT", "M1 · M2 · M3", "BATTER'S EYE", PARK_NAME, "HELD-OUT GAMES"];
  const mid = Math.floor(fascia.length / 2);
  words.forEach((w, k) => {
    const i = mid + (k - 2.5) * 9;
    const j = Math.round(i);
    plan.banners.push({ p: fascia[j], dir: [fascia[j + 1][0] - fascia[j - 1][0], fascia[j + 1][1] - fascia[j - 1][1]], text: w });
  });
  return plan;
}

/** Weighted choice: the first [value, weight] whose cumulative weight passes u (weights sum to 1). */
function pick(u, choices) {
  for (const [v, w] of choices) { if (u < w) return v; u -= w; }
  return choices.at(-1)[0];
}

/** One tree: conifers, round, cluster and poplar trees in the palette's greens, a few autumn. */
function tree(R, p, h) {
  const autumn = R() < 0.08;
  const greens = PALETTE.foliage;
  return {
    p, h, yaw: R() * 6.3, kind: pick(R(), [["conifer", 0.4], ["round", 0.22], ["cluster", 0.28], ["poplar", 0.1]]),
    color: autumn ? PALETTE.foliageAutumn[Math.floor(R() * 2)] : greens[Math.floor(R() * greens.length)],
  };
}

/** The whole starter park as one Group (scene feet). */
export function buildPark() {
  // Floodlights fully on inside 330 ft of the bowl's centre, gone by 560 (the island's trees);
  // the shadow box covers the stadium (~900 ft square, ~0.22 ft per shadow texel).
  const plan = makePlan();
  // Night pools: each tower's lights land 70% of the way to the field centre, and the infield
  // (where the pitcher and the plate are) gets its own, brightest-lit pool, as real parks aim it.
  const poolsFt = plan.towers.map((t) => [0, 1].map((j) => t.p[j] + 0.7 * (FIELD_CENTRE_FT[j] - t.p[j])).concat(135));
  poolsFt.push([0, 55, 150]);
  setParkLighting({ centreFt: FIELD_CENTRE_FT, floodRadiiFt: [330, 560], shadowHalfFt: 450, poolsFt });
  const g = new THREE.Group();
  g.name = "park";
  g.add(buildGround({ ...plan, trees: plan.groundTrees, props: [
    ...plan.umbrellas, ...plan.kiosks.map((k) => ({ ...k, r: 9 })), ...plan.fountains.map((p) => ({ p, r: 12 })),
    ...plan.tents.map((t) => ({ ...t, r: 8 })), ...plan.bushes.map((b) => ({ p: b.p, r: b.s * 0.9 })), { p: plan.gazebo, r: 15 },
  ].map((u) => ({ p: u.p, r: u.r ?? 5 })) }));
  g.add(buildStands());
  const flags = [], poles = [];
  const props = buildProps(plan, flags, poles);
  g.add(props, buildSurrounds(plan, flags, poles), ...buildFlags(flags, poles), buildBrand(plan));
  mergeStatic(g); // fewer draw calls (see build.js)
  // Everything animated registers a tick(t); collected once here, not traversed every frame.
  const tickers = [];
  g.traverse((o) => { if (o.userData.tick) tickers.push(o.userData.tick); });
  g.userData.update = ({ timeS, board }) => {
    for (const f of tickers) f(timeS);
    props.userData.setBoard(board);
  };
  return g;
}
