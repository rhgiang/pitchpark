import * as THREE from "three";
import { PALETTE } from "../look.js";
import { buildGround, offsetPath } from "./ground.js";
import { buildStands, mainPath, upperPath, bleacherPaths, MAIN_DEPTH_FT, BLEACHER_DEPTH_FT } from "./stands.js";
import { buildProps, ISLETS } from "./props.js";
import {
  ISLAND, fieldBoundary, fencePath, polar, fenceDistFt, resample, normals, rng, inside,
  BATTERS_EYE_HALF_DEG,
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
  };

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

  // Trees: rejection-sampled over the island, clear of the field, stands, plazas, paths, pond.
  const furniture = [...plan.umbrellas.map((u) => u.p), ...plan.kiosks.map((k) => k.p), ...plan.towers.map((t) => t.p)];
  const bleachAll = bleachers.flat();
  for (let tries = 0; tries < 4000 && plan.trees.length < 760; tries++) {
    const a = R() * 2 * Math.PI, r = Math.sqrt(R()) * (ISLAND.radiusFt - 14);
    const p = [cx + r * Math.cos(a), cy + r * Math.sin(a)];
    if (inside(p, boundary)) continue;
    if (distTo(p, main) < MAIN_DEPTH_FT + 52 || distTo(p, fence) < 26 + 32 + BLEACHER_DEPTH_FT) continue;
    if (distTo(p, bleachAll) < BLEACHER_DEPTH_FT + 30) continue;
    if (distTo(p, allPath) < 13 || distTo(p, furniture) < 14) continue;
    if (Math.hypot((p[0] - plan.pond.p[0]) / (plan.pond.rx + 10), (p[1] - plan.pond.p[1]) / (plan.pond.ry + 10)) < 1) continue;
    plan.trees.push(tree(R, p, 26 + R() * 30));
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
  const words = ["PITCHSIM", "STATCAST FIT", "M1 · M2 · M3", "BATTER'S EYE", "PITCHSIM", "HELD-OUT GAMES"];
  const mid = Math.floor(fascia.length / 2);
  words.forEach((w, k) => {
    const i = mid + (k - 2.5) * 9;
    const j = Math.round(i);
    plan.banners.push({ p: fascia[j], dir: [fascia[j + 1][0] - fascia[j - 1][0], fascia[j + 1][1] - fascia[j - 1][1]], text: w });
  });
  return plan;
}

/** One tree: mostly conifers and round trees in the palette's greens, a few autumn ones. */
function tree(R, p, h) {
  const autumn = R() < 0.08;
  const greens = PALETTE.foliage;
  return {
    p, h, yaw: R() * 6.3, kind: R() < 0.6 ? "conifer" : "round",
    color: autumn ? PALETTE.foliageAutumn[Math.floor(R() * 2)] : greens[Math.floor(R() * greens.length)],
  };
}

/** The whole starter park as one Group (scene feet). */
export function buildPark() {
  const plan = makePlan();
  const g = new THREE.Group();
  g.name = "park";
  g.add(buildGround({ ...plan, trees: plan.groundTrees, props: [...plan.umbrellas, ...plan.kiosks.map((k) => ({ ...k, r: 9 }))].map((u) => ({ p: u.p, r: u.r ?? 5 })) }));
  g.add(buildStands());
  g.add(buildProps(plan));
  return g;
}
