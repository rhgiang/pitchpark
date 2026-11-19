import * as THREE from "three";
import { mergeGeometries } from "three/examples/jsm/utils/BufferGeometryUtils.js";
import { PALETTE, makeToonMaterial, windowLitMaterial } from "../look.js";
import { box } from "./build.js";
import { drawWordmark } from "./brand.js";
import { polar, normals, BACKSTOP_FT } from "./layout.js";

/*
 * Behind home plate: the backdrop of the TV view (?old=home restores the single repeated banner).
 *
 * The TV camera (cameras.js) is a 2.6 degree telephoto from centre field: through the plate it sees
 * only about 21 ft of height, and behind home that is the padded wall (0 to 4.5 ft) and the first
 * row of seats. So that strip is the whole backdrop: this fills the wall with three rows of
 * different ads (a lower board row, a middle one and a ribbon board at the top), and the fan rail
 * and first seats above it get a rail and posts. Higher up, the press box hangs under the roof.
 *
 * All the ads share ONE atlas texture, ONE material and are merged into one mesh: one draw call.
 * The names are invented (Pitch Park themed; no real brands).
 */

export const AD_BRANDS = [
  // [name, tagline, background, text, accent, motif]
  ["CURVEBALL COFFEE", "bends toward the bean", "#f4ead2", "#2a4a6b", "#e5604b", "ring"],
  ["SEAMLESS DENIM", "no wake, no drag", "#2a4a6b", "#f4ead2", "#f2c14e", "stripes"],
  ["FASTBALL FUEL", "95 mph energy", "#f2c14e", "#2a4a6b", "#e5604b", "chevrons"],
  ["KNUCKLE BREWING", "unpredictable since 2026", "#3fa7a0", "#f4ead2", "#2a4a6b", "ring"],
  ["SLIDER SODA", "sweeps the shelf", "#e5604b", "#f4ead2", "#f2c14e", "stripes"],
  ["BULLPEN BURGERS", "always warm", "#f4ead2", "#e5604b", "#2a4a6b", "bars"],
  ["HOME STRETCH BANK", "run it out", "#2a4a6b", "#f2c14e", "#3fa7a0", "chevrons"],
  ["CHANGEUP AUTO", "same arm, new speed", "#ffffff", "#3fa7a0", "#e5604b", "bars"],
  ["RIVET RAMEN", "slurp season", "#d98aa0", "#2a4a6b", "#f4ead2", "ring"],
  ["SPINDRIFT SPORTS", "2,300 rpm of gear", "#7b52ab", "#f4ead2", "#f2c14e", "stripes"],
  ["GLOVE & GOLD", "oiled and ready", "#2f6b5a", "#f2c14e", "#f4ead2", "bars"],
  ["PITCH PARK", "", null, null, null, "wordmark"],
];
export const AD_TILE = { w: 1024, h: 256, cols: 2 };
const ROWS = Math.ceil(AD_BRANDS.length / AD_TILE.cols);

/** Tile rectangle of ad i in the atlas, in UV (v up): [u0, v0, u1, v1]. */
export function adUV(i) {
  const c = i % AD_TILE.cols, r = Math.floor(i / AD_TILE.cols);
  return [c / AD_TILE.cols, 1 - (r + 1) / ROWS, (c + 1) / AD_TILE.cols, 1 - r / ROWS];
}

/** Paint one ad (tile at the origin of g, w x h px): colour field, a motif on the left, name and tagline. */
export function drawAd(g, [name, tag, bg, fg, accent, motif], w, h) {
  if (motif === "wordmark") return drawWordmark(g, w, h);
  g.fillStyle = bg; g.fillRect(0, 0, w, h);
  g.fillStyle = accent;
  if (motif === "ring") { g.beginPath(); g.arc(h * 0.5, h / 2, h * 0.34, 0, 7); g.fill(); g.fillStyle = bg; g.beginPath(); g.arc(h * 0.5, h / 2, h * 0.17, 0, 7); g.fill(); }
  else if (motif === "stripes") for (let k = 0; k < 4; k++) g.fillRect(h * 0.1 + k * h * 0.2, 0, h * 0.09, h);
  else if (motif === "chevrons") for (let k = 0; k < 3; k++) { g.beginPath(); const x = h * 0.12 + k * h * 0.28; g.moveTo(x, h * 0.2); g.lineTo(x + h * 0.2, h / 2); g.lineTo(x, h * 0.8); g.lineTo(x + h * 0.09, h * 0.8); g.lineTo(x + h * 0.29, h / 2); g.lineTo(x + h * 0.09, h * 0.2); g.fill(); }
  else for (let k = 0; k < 4; k++) g.fillRect(h * 0.12 + k * h * 0.2, h * (0.7 - 0.15 * k), h * 0.12, h * (0.15 + 0.15 * k)); // rising bars
  const x0 = h * 1.05, avail = w - x0 - h * 0.12;
  g.fillStyle = fg; g.textBaseline = "middle"; g.textAlign = "left";
  g.font = "bold 100px sans-serif";
  const px = Math.min(h * 0.46, (avail * 100) / g.measureText(name).width);
  g.font = `bold ${px}px sans-serif`;
  g.fillText(name, x0, h * 0.42);
  g.font = `${px * 0.45}px sans-serif`;
  g.fillText(tag, x0, h * 0.78);
  g.strokeStyle = fg; g.lineWidth = h * 0.02; g.strokeRect(h * 0.05, h * 0.05, w - h * 0.1, h * 0.9);
}

let atlas = null;
function adAtlas() {
  if (atlas) return atlas;
  const c = document.createElement("canvas");
  c.width = AD_TILE.w * AD_TILE.cols; c.height = AD_TILE.h * ROWS;
  const g = c.getContext("2d");
  AD_BRANDS.forEach((b, i) => {
    g.save();
    g.translate((i % AD_TILE.cols) * AD_TILE.w, Math.floor(i / AD_TILE.cols) * AD_TILE.h);
    g.beginPath(); g.rect(0, 0, AD_TILE.w, AD_TILE.h); g.clip();
    drawAd(g, b, AD_TILE.w, AD_TILE.h);
    g.restore();
  });
  atlas = new THREE.CanvasTexture(c);
  atlas.colorSpace = THREE.SRGBColorSpace;
  atlas.anisotropy = 8;
  return atlas;
}

// The padded wall's rows: centre height and size of each (4:1 boards). 3 rows x 1.3 ft = the 4.5 ft wall.
export const AD_ROWS = [{ z: 0.8, w: 5.2 }, { z: 2.3, w: 5.2 }, { z: 3.75, w: 5.2 }];
export const WALL_R_FT = BACKSTOP_FT - 0.3; // a hair in front of the wall

/**
 * Where the ads go (Statcast ft): boards round the 60 ft arc behind home (135 to 225 deg), facing
 * home. Neighbours in a row, and the board above and below, never show the same ad.
 * @returns [{p: [x, y] centre, n: [nx, ny] unit normal, z, w, h, tile}]
 */
export function adLayout() {
  const out = [];
  const step = ((AD_ROWS[0].w + 0.25) / WALL_R_FT) * (180 / Math.PI); // degrees per board
  const n = Math.floor(90 / step);
  AD_ROWS.forEach(({ z, w }, row) => {
    for (let k = 0; k < n; k++) {
      const a = 180 + (k - (n - 1) / 2) * step, [x, y] = polar(a, WALL_R_FT);
      // Stride 5 (coprime with the 12 ads) along a row, shifted 4 per row (so diagonal neighbours differ too): no repeats close by.
      const tile = (k * 5 + row * 4 + 3) % AD_BRANDS.length;
      out.push({ p: [x, y], n: [-x / WALL_R_FT, -y / WALL_R_FT], z, w, h: w / 4, tile });
    }
  });
  return out;
}

/** The ad boards as one merged textured mesh. */
function adBoards() {
  const geos = adLayout().map((b) => {
    const g = new THREE.PlaneGeometry(b.w, b.h);
    const [u0, v0, u1, v1] = adUV(b.tile);
    const uv = g.attributes.uv;
    for (let i = 0; i < uv.count; i++) uv.setXY(i, uv.getX(i) ? u1 : u0, uv.getY(i) ? v1 : v0);
    g.rotateY(Math.atan2(b.n[0], -b.n[1])); // PlaneGeometry faces scene +z = Statcast -y
    g.translate(b.p[0], b.z, -b.p[1]);
    return g;
  });
  const tex = adAtlas();
  // Polygon offset: boards 0.3 ft off the wall seen from 450 ft are below the depth buffer's resolution.
  const mat = makeToonMaterial(0xffffff, { map: tex, emissiveMap: tex, emissive: 0xffffff, emissiveIntensity: 0.35, surface: "none",
    polygonOffset: true, polygonOffsetFactor: -2, polygonOffsetUnits: -8 });
  return new THREE.Mesh(mergeGeometries(geos), mat);
}

/** Fan rail along the top of the wall: a steel bar on posts every 6 ft, on the arc (instanced posts). */
function fanRail() {
  const g = new THREE.Group();
  const steel = makeToonMaterial(PALETTE.steelLight);
  const pts = [];
  for (let a = 135; a <= 225 + 1e-9; a += 2) pts.push(polar(a, BACKSTOP_FT + 1.3));
  const posts = new THREE.InstancedMesh(new THREE.BoxGeometry(0.16, 1.3, 0.16).translate(0, 0.65, 0), steel, pts.length);
  const m = new THREE.Matrix4();
  pts.forEach(([x, y], i) => posts.setMatrixAt(i, m.makeTranslation(x, 4.5, -y)));
  posts.castShadow = true;
  g.add(posts);
  // Top bar: one thin box per arc step.
  pts.slice(1).forEach(([x, y], i) => {
    const [x0, y0] = pts[i], len = Math.hypot(x - x0, y - y0);
    const bar = box(steel, len + 0.05, 0.14, 0.14, (x + x0) / 2, 5.7, -(y + y0) / 2);
    bar.rotation.y = Math.atan2(y - y0, x - x0);
    g.add(bar);
  });
  return g;
}

/**
 * Press box: a cream box with a long dark window band and a coral lid, hung under the main roof at
 * the back of the upper deck, centred behind home (it lights up at night like the other windows).
 * @param path the upper deck's path (stands.js upperPath)
 */
function pressBox(path) {
  const i = Math.floor(path.length / 2), n = normals(path)[i], s = 92;
  const g = new THREE.Group();
  g.position.set(path[i][0] + n[0] * s, 0, -(path[i][1] + n[1] * s));
  g.rotation.y = Math.atan2(n[0], -n[1]); // local +z points out, away from the field
  const cream = makeToonMaterial(PALETTE.cream), coral = makeToonMaterial(PALETTE.coral), steel = makeToonMaterial(PALETTE.steel);
  const W = 60, D = 12, Y0 = 72, H = 10.5;
  g.add(box(cream, W, H, D, 0, Y0, 0));
  g.add(box(coral, W + 1.2, 0.8, D + 1.2, 0, Y0 + H, 0));
  g.add(box(coral, W + 1.2, 0.6, 0.4, 0, Y0 - 0.2, -D / 2)); // sill
  const glass = box(windowLitMaterial, W - 6, 4.6, 0.4, 0, Y0 + 3.2, -D / 2 - 0.05);
  glass.castShadow = false;
  g.add(glass);
  for (let x = -W / 2 + 3; x <= W / 2 - 3 + 1e-6; x += 6) g.add(box(cream, 0.5, 4.6, 0.5, x, Y0 + 3.2, -D / 2 - 0.1)); // mullions
  for (const x of [-W / 2 + 2, W / 2 - 2]) g.add(box(steel, 1.2, Y0 - 67.6, 1.2, x, 67.6, D / 2 - 1.5)); // posts to the concourse
  return g;
}

/** Everything behind home: ad boards, fan rail, press box. */
export function buildHome(upperPath) {
  const g = new THREE.Group();
  g.add(adBoards(), fanRail(), pressBox(upperPath));
  return g;
}
