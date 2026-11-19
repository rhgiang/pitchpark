import * as THREE from "three";
import { mergeGeometries } from "three/examples/jsm/utils/BufferGeometryUtils.js";
import { PALETTE, makeToonMaterial } from "../look.js";
import { box } from "./build.js";
import { normals, polar, fenceDistFt } from "./layout.js";
import { isOld } from "./old.js";
import { scoreboardDeg } from "./signature.js";
import { upperPath, bleacherPaths, MAIN_DEPTH_FT, BLEACHER_DEPTH_FT } from "./stands.js";

/*
 * The park's name and its signs. To rename the park, change PARK_NAME; a different park module
 * can reuse wordmarkTexture() and call buildBrand() with its own sign list (brandSigns()).
 * All the big signs share ONE 4:1 texture and ONE material, and are merged into one mesh, so the
 * branding costs a single draw call; the text is drawn once at load (no per-frame work).
 */
export const PARK_NAME = "PITCH PARK";
const NAVY = "#1d3550", GOLD = "#f2c14e", CREAM = "#f4ead2"; // the start screen's colours (ui.css)
export const WORDMARK_PX = { w: 1024, h: 256 }; // 4:1, power of two: full mipmap chain, sharp from afar

/** Paint the wordmark: navy board, "PITCH" in white and "PARK" in gold (bold, as on the start screen), gold rule. */
export function drawWordmark(g, w, h) {
  g.fillStyle = NAVY; g.fillRect(0, 0, w, h);
  g.strokeStyle = CREAM; g.lineWidth = Math.max(2, h * 0.025);
  g.strokeRect(h * 0.06, h * 0.06, w - h * 0.12, h - h * 0.12);
  // Fit the text to the board: as big as 58% of the height, or 86% of the width, whichever is smaller.
  g.font = "bold 100px sans-serif"; g.textBaseline = "middle"; g.textAlign = "left";
  const [a, b] = PARK_NAME.split(" ");
  const wa = g.measureText(a + " ").width, wb = g.measureText(b).width;
  const px = Math.min(h * 0.58, (w * 0.86 * 100) / (wa + wb));
  g.font = `bold ${px}px sans-serif`;
  const k = px / 100, x0 = (w - (wa + wb) * k) / 2, y = h * 0.47;
  g.fillStyle = "#ffffff"; g.fillText(a, x0, y);
  g.fillStyle = GOLD; g.fillText(b, x0 + wa * k, y);
  g.fillRect(x0, h * 0.82, (wa + wb) * k, h * 0.04); // rule under the name
}

const cache = new Map();
/** Shared texture of the wordmark at w x h px (one per size). Aspect should be near 4:1; mipmapped. */
export function wordmarkTexture(w = WORDMARK_PX.w, h = WORDMARK_PX.h, doc = document) {
  const key = `${w}x${h}`;
  if (cache.has(key)) return cache.get(key);
  const c = doc.createElement("canvas");
  c.width = w; c.height = h;
  drawWordmark(c.getContext("2d"), w, h);
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  t.anisotropy = 8;
  cache.set(key, t);
  return t;
}

const unit = ([x, y]) => { const n = Math.hypot(x, y); return [x / n, y / n]; };

/**
 * Where the signs go (Statcast ft). Each: {id, p: [x, y] centre in plan, z: centre height, w, h (ft),
 * n: unit normal the face looks along, back: also a back-to-back face for free-standing signs}.
 * Nothing stands between the cameras and the pitcher, plate or ball (test/brand.test.js).
 */
export function brandSigns(plan) {
  const S = [];
  const add = (id, p, z, w, n, extra = {}) => S.push({ id, p, z, w, h: w / 4, n: unit(n), ...extra });
  // 1. Behind home plate, on the padded wall under the backstop net: the ad boards in the TV view
  //    (the camera sees that strip, 0 to 4.5 ft up, right above the plate). Faces home.
  for (let k = -4; k <= 4; k++) {
    const a = 180 + k * 7.2, [x, y] = polar(a, 59.7);
    add(`backstop${k}`, [x, y], 2.4, 7.3, [-x, -y]);
  }
  // 2. Scoreboard: on the cream block over the board (props.js scoreboard()), facing home.
  const sb = polar(scoreboardDeg(), fenceDistFt(scoreboardDeg()) + 32 + BLEACHER_DEPTH_FT + 20);
  const toHome = unit([-sb[0], -sb[1]]);
  add("scoreboard", [sb[0] + toHome[0] * 1.6, sb[1] + toHome[1] * 1.6], 90.5, 28, toHome);
  // 3. Main grandstand: a big rooftop sign (on posts, over the roof, facing out behind home) and
  //    three panels on the outer back wall, between the window rows and the roof.
  const up = upperPath(), un = normals(up);
  const mid = Math.floor(up.length / 2);
  const at = (i, s) => [up[i][0] + un[i][0] * s, up[i][1] + un[i][1] * s];
  add("roof", at(mid, 92), 106, 96, un[mid], { roof: true, back: true }); // two faces: the field side shows in the intro
  for (const d of [-26, 0, 26]) add(`facade${d}`, at(mid + d, MAIN_DEPTH_FT + 1), 69, 44, un[mid + d]);
  // 4. Outfield bleachers' back walls, one each.
  for (const b of bleacherPaths()) {
    const bn = normals(b), i = Math.floor(b.length / 2);
    add("bleacher", [b[i][0] + bn[i][0] * (BLEACHER_DEPTH_FT + 0.4), b[i][1] + bn[i][1] * (BLEACHER_DEPTH_FT + 0.4)], 26, 22, bn[i]);
  }
  // 5. Plaza: on each kiosk roof (two faces), and two entrance boards flanking the gate.
  for (const k of plan.kiosks) {
    const t = k.yaw; // kiosk front (local -z) looks along (-sin yaw, cos yaw)
    add("kiosk", k.p, 11.7, 10, [-Math.sin(t), Math.cos(t)], { back: true, post: 10.2 });
  }
  for (const s of [-1, 1]) add("gate", [plan.gate.p[0] + s * 70, plan.gate.p[1] - 20], 15, 22, [0, -1], { back: true, post: 0 });
  add("lot", [plan.lot.c[0] - plan.lot.w / 2 - 6, plan.lot.c[1]], 15, 22, [-1, 0], { back: true, post: 0 });
  return S;
}

/** One sign's plane(s) in scene space: normal n (Statcast), centre (x, z, -y). */
function signGeometry(s) {
  const faces = s.back ? [1, -1] : [1];
  return faces.map((f) => {
    const g = new THREE.PlaneGeometry(s.w, s.h);
    // PlaneGeometry faces scene +z, which is Statcast -y; yaw it round to the wanted normal.
    g.rotateY(Math.atan2(f * s.n[0], -f * s.n[1]));
    g.translate(s.p[0] + (s.back ? 0.15 * f * s.n[0] : 0), s.z, -(s.p[1] + (s.back ? 0.15 * f * s.n[1] : 0)));
    return g;
  });
}

/**
 * All the signs: one merged textured mesh, plus plain posts and the rooftop sign's backing
 * (merged with the park's other static boxes by mergeStatic).
 */
export function buildBrand(plan) {
  const g = new THREE.Group();
  const tex = wordmarkTexture();
  // The new backdrop behind home (home.js) has its own ads; ?old=home keeps the nine wordmark boards.
  const signs = brandSigns(plan).filter((x) => isOld("home") || !x.id.startsWith("backstop"));
  // Lit like the scoreboard (emissive map): readable at night, and the gold blooms a little.
  // Polygon offset: a sign 0.2 ft off a wall seen from 1400 ft is below the depth buffer's resolution.
  const mat = makeToonMaterial(0xffffff, { map: tex, emissiveMap: tex, emissive: 0xffffff, emissiveIntensity: 0.35, surface: "none",
    polygonOffset: true, polygonOffsetFactor: -2, polygonOffsetUnits: -8 });
  g.add(new THREE.Mesh(mergeGeometries(signs.flatMap(signGeometry)), mat));
  const steel = makeToonMaterial(PALETTE.steel), coral = makeToonMaterial(PALETTE.coral);
  for (const s of signs) {
    const f = new THREE.Group(); // local x along the sign, z along its normal (scene)
    f.position.set(s.p[0], 0, -s.p[1]);
    f.rotation.y = Math.atan2(s.n[0], -s.n[1]);
    if (s.roof) { // coral frame slab between the two faces, on four posts standing on the roof
      f.add(box(coral, s.w + 3, s.h + 3, 0.28, 0, s.z - s.h / 2 - 1.5, 0));
      for (const x of [-s.w * 0.4, -s.w * 0.13, s.w * 0.13, s.w * 0.4]) f.add(box(steel, 1.6, 12, 1.6, x, 86, 0));
    } else if (s.post != null) { // free-standing: two posts under the sign
      for (const x of [-s.w * 0.4, s.w * 0.4]) f.add(box(steel, 0.8, s.z - s.h / 2 - s.post, 0.8, x, s.post, 0));
    }
    g.add(f);
  }
  return g;
}
