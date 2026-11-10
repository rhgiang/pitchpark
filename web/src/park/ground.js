import * as THREE from "three";
import { PALETTE, makeToonMaterial } from "../look.js";
import {
  BASES, BASE_SIZE_FT, MOUND_CENTRE_Y_FT, ISLAND, fieldBoundary, polar,
  normals, BLEACHER_GAP_FT, BATTERS_EYE_HALF_DEG, fenceDistFt, rng,
} from "./layout.js";

// The ground is painted, not modelled: one canvas painter draws grass stripes, dirt, chalk,
// paths and baked contact shadows in Statcast feet, and is run three times at different scales
// (whole island, field, home plate area) so the chalk near the batter stays sharp. Painting
// shadows into the texture is the code version of the baked lightmaps the references use.

const hex = (c) => `#${new THREE.Color(c).getHexString()}`;
const IN = 1 / 12;

/** Offset a polyline along its outward normals by d ft (d can vary per point via a function). */
export function offsetPath(pts, d) {
  const n = normals(pts);
  return pts.map((p, i) => [p[0] + n[i][0] * (typeof d === "function" ? d(i) : d), p[1] + n[i][1] * (typeof d === "function" ? d(i) : d)]);
}

function poly(ctx, pts, close = true) {
  ctx.beginPath();
  pts.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
  if (close) ctx.closePath();
}

function line(ctx, a, b, w, color = PALETTE.chalk) {
  ctx.strokeStyle = hex(color);
  ctx.lineWidth = w;
  ctx.lineCap = "butt";
  poly(ctx, [a, b], false);
  ctx.stroke();
}

function rect(ctx, x0, y0, x1, y1, w) {
  ctx.strokeStyle = hex(PALETTE.chalk);
  ctx.lineWidth = w;
  ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
}

function disc(ctx, [x, y], r, color) {
  ctx.fillStyle = typeof color === "string" ? color : hex(color);
  ctx.beginPath();
  ctx.arc(x, y, r, 0, 2 * Math.PI);
  ctx.fill();
}

/** Soft dark blob (a baked contact shadow): radial gradient from `a` alpha to 0. */
function blob(ctx, [x, y], r, a = 0.35) {
  const g = ctx.createRadialGradient(x, y, 0, x, y, r);
  g.addColorStop(0, `rgba(20,40,30,${a})`);
  g.addColorStop(1, "rgba(20,40,30,0)");
  ctx.fillStyle = g;
  ctx.fillRect(x - r, y - r, 2 * r, 2 * r);
}

/** Soft darkening along a polyline: stacked translucent strokes, widest faintest. */
function contactShade(ctx, pts, widthFt, a = 0.05, close = false) {
  ctx.lineJoin = "round";
  for (let w = widthFt; w > 0.5; w *= 0.7) {
    ctx.strokeStyle = `rgba(20,40,30,${a})`;
    ctx.lineWidth = w;
    poly(ctx, pts, close);
    ctx.stroke();
  }
}

/**
 * Paint everything on the ground into ctx, whose transform maps Statcast feet to pixels.
 * @param plan {{trees: {p: number[], r: number}[], props: {p: number[], r: number}[],
 *   mainOuter: number[][], bleacherOuter: number[][][]}}
 */
function paint(ctx, plan) {
  const R = rng(7);
  const [cx, cy] = ISLAND.centre;
  // Island grass, with darker and lighter patches.
  disc(ctx, ISLAND.centre, ISLAND.radiusFt + 5, PALETTE.grassDeep);
  for (let i = 0; i < 260; i++) {
    const a = R() * 2 * Math.PI, r = Math.sqrt(R()) * ISLAND.radiusFt;
    ctx.globalAlpha = 0.18;
    disc(ctx, [cx + r * Math.cos(a), cy + r * Math.sin(a)], 8 + R() * 30,
      R() < 0.5 ? PALETTE.foliage[0] : PALETTE.grassMow);
  }
  ctx.globalAlpha = 1;

  // Plaza: paving round the outside of the stands, and paths out to the island's edge.
  ctx.lineJoin = "round";
  ctx.lineCap = "round";
  ctx.strokeStyle = hex(PALETTE.paving);
  ctx.lineWidth = 70;
  poly(ctx, plan.mainOuter, false); ctx.stroke();
  for (const b of plan.bleacherOuter) { ctx.lineWidth = 46; poly(ctx, b, false); ctx.stroke(); }
  ctx.lineWidth = 16;
  for (const p of plan.paths) { poly(ctx, p, false); ctx.stroke(); }
  // Pond, behind the left-field stands: water with a lighter shallow rim.
  const { p: pond, rx, ry } = plan.pond;
  ctx.fillStyle = hex(PALETTE.waterLight);
  ctx.beginPath(); ctx.ellipse(...pond, rx, ry, 0, 0, 2 * Math.PI); ctx.fill();
  ctx.fillStyle = hex(PALETTE.water);
  ctx.beginPath(); ctx.ellipse(...pond, rx - 7, ry - 7, 0, 0, 2 * Math.PI); ctx.fill();

  // Baked contact shadows: round the stands and under every tree and prop.
  contactShade(ctx, plan.mainOuter, 24, 0.06);
  for (const b of plan.bleacherOuter) contactShade(ctx, b, 20, 0.06);
  for (const t of plan.trees) blob(ctx, t.p, t.h * 0.35, 0.4);
  for (const t of plan.props) blob(ctx, t.p, t.r * 1.5, 0.3);

  // ---- the playing field, clipped to the walls ----
  const boundary = fieldBoundary();
  ctx.save();
  poly(ctx, boundary); ctx.clip();
  ctx.fillStyle = hex(PALETTE.grass);
  ctx.fillRect(-500, -200, 1000, 800);
  // Mowing: 15 ft stripes across the centre-field line in the outfield, a checkerboard on the
  // infield grass (both common MLB patterns).
  ctx.fillStyle = hex(PALETTE.grassMow);
  for (let y = 0; y < 460; y += 30) ctx.fillRect(-400, y + 15, 800, 15);
  // Foul territory (|x| > y): plain, unstriped grass.
  ctx.fillStyle = hex(PALETTE.grassFoul);
  poly(ctx, [[0, 0], [500, 500], [500, -500], [-500, -500], [-500, 500]]); ctx.fill();

  // Warning track: 15 ft of dirt inside every wall.
  ctx.strokeStyle = hex(PALETTE.dirtDark);
  ctx.lineWidth = 30;
  ctx.lineJoin = "round";
  poly(ctx, boundary); ctx.stroke();
  // Dirt behind home plate, up to the backstop.
  ctx.fillStyle = hex(PALETTE.dirt);
  ctx.beginPath(); ctx.arc(0, 0, 60, Math.PI * 1.25, Math.PI * 1.75); ctx.lineTo(0, 0); ctx.closePath(); ctx.fill();

  // Infield dirt: everything within 95 ft of the front of the rubber (60.5 ft), between the
  // foul lines (with 4 ft of dirt outside them for the base paths).
  ctx.fillStyle = hex(PALETTE.dirt);
  ctx.save();
  const o = 4 * Math.SQRT1_2; // 4 ft off a 45 deg line
  poly(ctx, [[0, -6], [-141.4 - o, 141.4 - o], [0, 300], [141.4 + o, 141.4 - o]]); ctx.clip();
  disc(ctx, [0, 60.5], 95, PALETTE.dirt);
  ctx.restore();
  // Infield grass: the diamond inside the base paths (3 ft each side of the base lines).
  const k = 3 * Math.SQRT2;
  ctx.fillStyle = hex(PALETTE.grass);
  poly(ctx, [[0, k], [BASES.first[0] - k, BASES.first[1]], [0, BASES.second[1] - k], [BASES.third[0] + k, BASES.third[1]]]);
  ctx.fill();
  ctx.save(); ctx.clip();
  ctx.rotate(Math.PI / 4);
  ctx.fillStyle = hex(PALETTE.grassMow);
  for (let i = 0; i < 8; i++) for (let j = 0; j < 8; j++) if ((i + j) % 2) ctx.fillRect(i * 11.25, j * 11.25, 11.25, 11.25);
  ctx.restore();
  // Dirt circles: home plate (26 ft across, OBR Appendix 2) and the mound (18 ft).
  disc(ctx, [0, 0.7], 13, PALETTE.dirt);
  disc(ctx, [0, MOUND_CENTRE_Y_FT], 9, PALETTE.dirtLight);
  // Grass edges get a thin darker lip, as a lip of grass casts on the dirt.
  ctx.strokeStyle = "rgba(60,40,20,0.18)";
  ctx.lineWidth = 0.6;
  ctx.beginPath(); ctx.arc(0, 0.7, 13, 0, 2 * Math.PI); ctx.stroke();

  // On-deck circles (5 ft across), between home and the dugouts.
  for (const s of [-1, 1]) {
    disc(ctx, [s * 37, -8], 2.6, PALETTE.dirtDark);
    disc(ctx, [s * 37, -8], 2.3, PALETTE.teal);
  }

  // ---- chalk (3-4 in lines) ----
  const W = 4 * IN;
  // Foul lines from where they leave the batter's boxes to the poles.
  for (const s of [-1, 1]) line(ctx, polar(s * 45, 5.25), polar(s * 45, fenceDistFt(45)), W);
  // Batter's boxes: 4 x 6 ft, inner line 6 in from the plate, centred on the plate's middle.
  for (const s of [-1, 1]) rect(ctx, s > 0 ? 1.208 : -5.208, -2.29, s > 0 ? 5.208 : -1.208, 3.71, 3 * IN);
  // Catcher's box: 43 in wide, 8 ft back from the batter's boxes.
  const cw = 21.5 * IN;
  line(ctx, [-cw, -2.29], [-cw, -10.29], 3 * IN); line(ctx, [cw, -2.29], [cw, -10.29], 3 * IN);
  line(ctx, [-cw, -10.29], [cw, -10.29], 3 * IN);
  // Running lane: the last 45 ft to first base, 3 ft outside the line.
  const lane0 = polar(45, 45), lane1 = polar(45, 90);
  const off = [3 * Math.SQRT1_2, -3 * Math.SQRT1_2];
  line(ctx, [lane0[0] + off[0], lane0[1] + off[1]], [lane1[0] + off[0], lane1[1] + off[1]], W);
  line(ctx, lane0, [lane0[0] + off[0], lane0[1] + off[1]], W);
  // Coach's boxes: 20 x 10 ft, 15 ft outside the base lines (art: approximate placement).
  for (const s of [-1, 1]) {
    const at = (a, b) => [Math.SQRT1_2 * s * (a + b), Math.SQRT1_2 * (a - b)]; // a along, b out
    ctx.strokeStyle = hex(PALETTE.chalk); ctx.lineWidth = W;
    poly(ctx, [at(60, 15), at(80, 15), at(80, 25), at(60, 25)]); ctx.stroke();
  }

  // Contact shading at the foot of every wall, and a soft ring round the mound and bases.
  contactShade(ctx, boundary, 14, 0.07, true);
  blob(ctx, [0, MOUND_CENTRE_Y_FT], 12, 0.12);
  for (const b of Object.values(BASES)) blob(ctx, b, 2.2, 0.3);
  ctx.restore();

  // Bullpens in the gap behind the outfield fence: dirt strips with two mounds each.
  for (const s of [-1, 1]) {
    const pen = [];
    for (let a = 26; a <= 40; a += 1) pen.push(polar(s * a, fenceDistFt(a) + BLEACHER_GAP_FT / 2));
    ctx.strokeStyle = hex(PALETTE.dirt);
    ctx.lineWidth = BLEACHER_GAP_FT - 6;
    ctx.lineJoin = "round";
    poly(ctx, pen, false); ctx.stroke();
  }
  // Batter's eye planting strip in centre field.
  const eye = [];
  for (let a = -BATTERS_EYE_HALF_DEG; a <= BATTERS_EYE_HALF_DEG; a++) eye.push(polar(a, fenceDistFt(a) + 14));
  ctx.strokeStyle = hex(PALETTE.foliage[1]);
  ctx.lineWidth = 22;
  poly(ctx, eye, false); ctx.stroke();
}

/**
 * One painted ground layer: a canvas covering Statcast x0..x1, y0..y1 at `px` pixels on its
 * long side, on a plane (or disc) at y = 0. `order` pulls nearer layers in front in the depth
 * test (polygon offset), so the sharper home-plate layer wins over the field one beneath it.
 */
function layer(plan, [x0, x1, y0, y1], px, order, disc = false) {
  const w = x1 - x0, h = y1 - y0;
  const c = document.createElement("canvas");
  c.width = Math.round(w >= h ? px : (px * w) / h);
  c.height = Math.round(h >= w ? px : (px * h) / w);
  const ctx = c.getContext("2d");
  // Statcast (x, y) -> canvas pixels: x right, y up (canvas y runs down).
  ctx.setTransform(c.width / w, 0, 0, -c.height / h, (-x0 * c.width) / w, (y1 * c.height) / h);
  paint(ctx, plan);
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.anisotropy = 8;
  const m = makeToonMaterial(0xffffff, {
    map: tex, polygonOffset: order > 0, polygonOffsetFactor: -order, polygonOffsetUnits: -2 * order,
  });
  const geo = disc ? new THREE.CircleGeometry(w / 2, 96) : new THREE.PlaneGeometry(w, h);
  const mesh = new THREE.Mesh(geo, m);
  mesh.rotation.x = -Math.PI / 2; // plane's +y -> scene -z = Statcast +y
  mesh.position.set((x0 + x1) / 2, 0, -(y0 + y1) / 2);
  mesh.receiveShadow = true;
  return mesh;
}

/**
 * The three bases: 18 in square, 3 in tall (OBR 2.03; 18 in since 2023). First and third lie
 * wholly in fair territory with their outer corner on the 90 ft point; second is centred on it.
 */
function bases() {
  const g = new THREE.Group();
  const m = makeToonMaterial(PALETTE.chalk);
  const s = BASE_SIZE_FT;
  const at = { first: [BASES.first[0] - s / Math.SQRT2, BASES.first[1]], second: BASES.second,
    third: [BASES.third[0] + s / Math.SQRT2, BASES.third[1]] };
  for (const [x, y] of Object.values(at)) {
    const b = new THREE.Mesh(new THREE.BoxGeometry(s, 0.25, s), m);
    b.position.set(x, 0.125, -y);
    b.rotation.y = Math.PI / 4;
    b.castShadow = b.receiveShadow = true;
    g.add(b);
  }
  return g;
}

/** The painted ground (island, field, home-plate area) and the bases. */
export function buildGround(plan) {
  const g = new THREE.Group();
  g.add(bases());
  const [cx, cy] = ISLAND.centre, r = ISLAND.radiusFt;
  g.add(layer(plan, [cx - r, cx + r, cy - r, cy + r], 2048, 0, true));
  g.add(layer(plan, [-340, 340, -110, 500], 4096, 1));
  g.add(layer(plan, [-16, 16, -13, 19], 2048, 2));
  return g;
}


