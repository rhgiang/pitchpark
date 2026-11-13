import * as THREE from "three";
import { PALETTE, makeToonMaterial, terrain } from "../look.js";
import {
  BASES, BASE_SIZE_FT, MOUND_CENTRE_Y_FT, ISLAND, fieldBoundary, polar,
  normals, BLEACHER_GAP_FT, BATTERS_EYE_HALF_DEG, fenceDistFt, rng, resample,
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
 * Smooth 1-D noise of arc length s (ft), roughly in [-1, 1]: three sines with seeded phases,
 * wavelengths 7.3, 2.9 and 1.3 ft. Edges cut by a groundskeeper wander at about these scales.
 */
function wobble(R) {
  const w = [[7.3, 0.5], [2.9, 0.32], [1.3, 0.18]].map(([L, a]) => [(2 * Math.PI) / L, a, R() * 2 * Math.PI]);
  return (s) => w.reduce((v, [f, a, p]) => v + a * Math.sin(f * s + p), 0);
}

/**
 * A closed outline made irregular: resampled every 0.25 ft and pushed along its normals by up to
 * `amp` ft. Grass/dirt edges are cut by hand and trodden, never perfect circles; the wobble
 * averages to zero, so the shapes keep their real sizes.
 */
function ragged(pts, amp, R) {
  const q = resample([...pts, pts[0]], 0.25);
  const n = normals(q), w = wobble(R);
  return q.map((p, i) => [p[0] + n[i][0] * amp * w(i * 0.25), p[1] + n[i][1] * amp * w(i * 0.25)]);
}

/** Points round a circle (Statcast ft), about every 0.5 ft. */
function circle([x, y], r) {
  const n = Math.max(24, Math.ceil((2 * Math.PI * r) / 0.5));
  return Array.from({ length: n }, (_, i) => [x + r * Math.cos((2 * Math.PI * i) / n), y + r * Math.sin((2 * Math.PI * i) / n)]);
}

/**
 * The lip where grass meets clay, in place of an ink line. On the clay side, a band of darker,
 * damper clay ~1.5 ft wide: infield mix washes and is swept against the grass edge and builds up
 * there (groundskeepers call it the lip). On the grass side, a narrow darker shade: the edge
 * turf is thicker and shades itself. `dirtInside`: whether the clay is inside the outline.
 */
function lip(ctx, pts, dirtInside) {
  ctx.lineJoin = "round";
  for (const [clay, bands] of [[true, [[3.2, 0.07], [1.8, 0.09], [0.7, 0.1]]], [false, [[1.4, 0.07], [0.6, 0.07]]]]) {
    ctx.save();
    poly(ctx, pts);
    if (clay === dirtInside) ctx.clip();
    else { ctx.rect(-3000, -3000, 6000, 6000); ctx.clip("evenodd"); } // everything outside pts
    for (const [w, a] of bands) { // soft: widest faintest
      ctx.strokeStyle = clay ? `rgba(110,48,22,${a})` : `rgba(35,55,20,${a})`;
      ctx.lineWidth = w;
      poly(ctx, pts); ctx.stroke();
    }
    ctx.restore();
  }
}

/**
 * Worn turf: many small translucent straw and bare-soil dots scattered round `c` (ellipse radii
 * rx, ry ft), denser in the middle, where players stand and walk.
 */
function wear(ctx, R, [x, y], rx, ry, n, a = 0.07) {
  for (let i = 0; i < n; i++) {
    const t = R() * 2 * Math.PI, r = Math.sqrt(R()) * R(); // concentrated toward the centre
    ctx.globalAlpha = a * (0.5 + R());
    disc(ctx, [x + Math.cos(t) * r * rx, y + Math.sin(t) * r * ry], 0.2 + R() * 0.8, R() < 0.8 ? 0xb9b064 : 0xa88a5a);
  }
  ctx.globalAlpha = 1;
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

  // Worn paving: darker and lighter flagstone patches and a few hairline cracks over the plaza.
  for (const q of plan.mainOuter) {
    if (R() < 0.5) continue;
    ctx.globalAlpha = 0.12;
    disc(ctx, [q[0] + (R() - 0.5) * 50, q[1] + (R() - 0.5) * 50], 3 + R() * 7, R() < 0.5 ? PALETTE.concreteShade : PALETTE.cream);
    ctx.globalAlpha = 0.25;
    ctx.strokeStyle = hex(PALETTE.concreteShade); ctx.lineWidth = 0.35;
    const c0 = [q[0] + (R() - 0.5) * 50, q[1] + (R() - 0.5) * 50];
    poly(ctx, [c0, [c0[0] + (R() - 0.5) * 8, c0[1] + (R() - 0.5) * 8], [c0[0] + (R() - 0.5) * 14, c0[1] + (R() - 0.5) * 14]], false);
    ctx.stroke();
  }
  ctx.globalAlpha = 1;
  // Gate forecourt, the path to the gazebo, the car park (asphalt with bay lines) and the stream.
  if (plan.gate) {
    ctx.fillStyle = hex(PALETTE.paving);
    ctx.fillRect(plan.gate.p[0] - 60, plan.gate.p[1] - 60, 120, 70);
    ctx.lineWidth = 16; ctx.strokeStyle = hex(PALETTE.paving);
    poly(ctx, [plan.pond.p, plan.gazebo], false); ctx.stroke();
    disc(ctx, plan.gazebo, 24, PALETTE.paving);
    const { c, w, h } = plan.lot;
    ctx.fillStyle = "#8d8f8a";
    ctx.fillRect(c[0] - w / 2, c[1] - h / 2, w, h);
    ctx.strokeStyle = hex(PALETTE.chalk); ctx.lineWidth = 0.4;
    for (let r = 0; r < 4; r++) for (let k = 0; k <= 7; k++) {
      const x = c[0] - w / 2 + 4 + k * 16, y = c[1] - h / 2 + 2 + r * 19;
      poly(ctx, [[x, y], [x, y + 15]], false); ctx.stroke();
    }
    ctx.lineCap = "round"; ctx.lineJoin = "round";
    ctx.strokeStyle = hex(PALETTE.waterLight); ctx.lineWidth = 15;
    poly(ctx, plan.stream, false); ctx.stroke();
    ctx.strokeStyle = hex(PALETTE.water); ctx.lineWidth = 9;
    poly(ctx, plan.stream, false); ctx.stroke();
    // Flower beds: clusters of coral, mustard, cream and pink dots round the fountains and gazebo.
    const flowers = [PALETTE.coral, PALETTE.mustard, PALETTE.cream, 0xd98aa0];
    for (const centre of [...plan.fountains, plan.gazebo, plan.gate.p]) {
      for (let i = 0; i < 160; i++) {
        const a = R() * 2 * Math.PI, r = 17 + R() * 8;
        disc(ctx, [centre[0] + Math.cos(a) * r, centre[1] + Math.sin(a) * r * (centre === plan.gate.p ? 0.4 : 1) - (centre === plan.gate.p ? 52 : 0)], 0.6 + R() * 0.6, flowers[Math.floor(R() * 4)]);
      }
    }
  }

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

  // Warning track: 15 ft of dirt inside every wall (a 30 ft stroke on the wall line, clipped).
  // Its grass edge gets the same lip as the infield: a shade on the grass side, painted under a
  // wider stroke, and darker clay along the track's inner edge, painted over and trimmed back.
  ctx.lineJoin = "round";
  for (const [style, w] of [["rgba(35,55,20,0.12)", 33], [hex(PALETTE.dirtDark), 30], ["rgba(110,48,22,0.14)", 30], [hex(PALETTE.dirtDark), 27]]) {
    ctx.strokeStyle = style; ctx.lineWidth = w;
    poly(ctx, boundary); ctx.stroke();
  }
  // Dirt behind home plate, up to the backstop.
  ctx.fillStyle = hex(PALETTE.dirt);
  ctx.beginPath(); ctx.arc(0, 0, 60, Math.PI * 1.25, Math.PI * 1.75); ctx.lineTo(0, 0); ctx.closePath(); ctx.fill();

  // Infield dirt: everything within 95 ft of the front of the rubber (60.5 ft), between the
  // foul lines (with 4 ft of dirt outside them for the base paths). Edges are ragged (see ragged).
  const o = 4 * Math.SQRT1_2; // 4 ft off a 45 deg line
  const skinClip = ragged([[0, -6], [-141.4 - o, 141.4 - o], [0, 300], [141.4 + o, 141.4 - o]], 0.15, R);
  const skinArc = ragged(circle([0, 60.5], 95), 0.2, R);
  ctx.fillStyle = hex(PALETTE.dirt);
  ctx.save(); poly(ctx, skinClip); ctx.clip(); poly(ctx, skinArc); ctx.fill(); ctx.restore();
  // Infield grass: the diamond inside the base paths (3 ft each side of the base lines).
  const k = 3 * Math.SQRT2;
  const diamond = ragged([[0, k], [BASES.first[0] - k, BASES.first[1]], [0, BASES.second[1] - k], [BASES.third[0] + k, BASES.third[1]]], 0.15, R);
  ctx.fillStyle = hex(PALETTE.grass);
  poly(ctx, diamond); ctx.fill();
  ctx.save(); ctx.clip();
  ctx.rotate(Math.PI / 4);
  ctx.fillStyle = hex(PALETTE.grassMow);
  for (let i = 0; i < 8; i++) for (let j = 0; j < 8; j++) if ((i + j) % 2) ctx.fillRect(i * 11.25, j * 11.25, 11.25, 11.25);
  ctx.restore();
  // Worn turf: the outfielders' spots, the coaches' boxes and where the corner infielders creep
  // onto the grass.
  for (const p of [[-150, 265], [0, 305], [150, 265]]) wear(ctx, R, p, 12, 9, 420);
  for (const s of [-1, 1]) {
    wear(ctx, R, [s * 63.6, 35.4], 9, 9, 260, 0.1);
    wear(ctx, R, [s * 52, 70], 6, 6, 120);
  }
  // Lips where grass meets clay, in place of ink (see lip). The home circle covers where the
  // diamond's and base paths' edges run into it, so it is painted after these.
  lip(ctx, diamond, false);
  ctx.save(); poly(ctx, skinClip); ctx.clip(); lip(ctx, skinArc, true); ctx.restore();
  ctx.save(); poly(ctx, skinArc); ctx.clip(); lip(ctx, skinClip, true); ctx.restore();
  // Dirt circles: home plate (26 ft across, OBR Appendix 2) and the mound (18 ft; the painted
  // clay runs a few inches past the modelled mound's foot, so no grass shows under its rim).
  const homeCircle = ragged(circle([0, 0.7], 13), 0.15, R);
  const moundCircle = ragged(circle([0, MOUND_CENTRE_Y_FT], 9.25), 0.12, R);
  ctx.fillStyle = hex(PALETTE.dirt); poly(ctx, homeCircle); ctx.fill();
  ctx.fillStyle = hex(PALETTE.dirtLight); poly(ctx, moundCircle); ctx.fill();
  for (const pts of [homeCircle, moundCircle]) lip(ctx, pts, true);

  // Damp, darker clay hugging home plate (watered and tamped round its edges) and the rubber.
  contactShade(ctx, [[-8.5 * IN, 17 * IN], [8.5 * IN, 17 * IN], [8.5 * IN, 8.5 * IN], [0, 0], [-8.5 * IN, 8.5 * IN]], 0.9, 0.06, true);
  contactShade(ctx, [[-1, 60.5], [1, 60.5], [1, 61], [-1, 61]], 0.9, 0.06, true);

  // On-deck circles (5 ft across), between home and the dugouts.
  for (const s of [-1, 1]) {
    disc(ctx, [s * 37, -8], 2.6, PALETTE.dirtDark);
    disc(ctx, [s * 37, -8], 2.3, PALETTE.teal);
  }

  // Chalk lines are not painted here: the ground shader draws them (park/layout.js chalkSegments).

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
    surface: "ground", map: tex, polygonOffset: order > 0, polygonOffsetFactor: -order, polygonOffsetUnits: -2 * order,
  });
  // Placed in the geometry, with the mesh at the origin. All three layers are terrain (one shared
  // ID in the outline pre-pass, look.js), so where they overlap they draw no false ink.
  const geo = (disc ? new THREE.CircleGeometry(w / 2, 96) : new THREE.PlaneGeometry(w, h))
    .rotateX(-Math.PI / 2) // plane's +y -> scene -z = Statcast +y
    .translate((x0 + x1) / 2, 0, -(y0 + y1) / 2);
  const mesh = terrain(new THREE.Mesh(geo, m));
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


