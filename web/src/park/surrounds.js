import * as THREE from "three";
import { mergeGeometries } from "three/examples/jsm/utils/BufferGeometryUtils.js";
import {
  PALETTE, makeToonMaterial, lampMaterial, noOutline, windowMaterial, windowLitMaterial,
} from "../look.js";
import { frameAt, box, place, shadeOf } from "./build.js";
import { rng } from "./layout.js";
import { wordmarkTexture } from "./brand.js";

/*
 * The island around the stadium (art pass 2): what makes the cricket clip's island read as a
 * place rather than a forest is variety at every scale, so this adds a grand gate building with a
 * clock tower behind home, fountains, a gazebo by the pond, food tents, a car park, lamp posts and
 * benches along the paths, bushes and boulders, a waterfall off the island's edge and rocks hanging
 * under it. Placements come from the plan (park/index.js); everything repeated is instanced.
 * Animated bits (flags, waterfall, clock) are advanced by update() from the park's update hook.
 */

const toon = (hex, surface) => makeToonMaterial(hex, surface ? { surface } : {});
const at = ([x, y], h = 0) => [x, h, -y];

/** Instanced mesh of `geo` at plan sites [{p, yaw, s, color, y}], shadows on. */
function instances(geo, mat, sites, { cast = true } = {}) {
  const m = new THREE.InstancedMesh(geo, mat, sites.length);
  const c = new THREE.Color();
  sites.forEach((q, i) => {
    place(m, i, at(q.p, q.y ?? 0), q.yaw ?? 0, q.s ?? 1);
    if (q.color !== undefined) m.setColorAt(i, c.set(q.color));
  });
  m.castShadow = cast;
  m.receiveShadow = true;
  return m;
}

/** A round-topped arch opening (6 x 7 ft, radius-3 top), 1.2 ft deep, base at y = 0. */
export function archGeometry(w = 6, h = 7, d = 1.2) {
  return mergeGeometries([
    new THREE.BoxGeometry(w, h, d).translate(0, h / 2, 0),
    new THREE.CylinderGeometry(w / 2, w / 2, d, 12, 1, false, 0, Math.PI).rotateX(Math.PI / 2).rotateZ(Math.PI / 2).translate(0, h, 0),
  ].map((g) => g.toNonIndexed()));
}

// ---------- waving flags (CPU-animated so the outline pre-pass sees the same shape) ----------

/**
 * All the park's flags as one mesh whose vertices are moved each frame: a travelling sine wave
 * along each flag, growing from the pole to the tip (a flag is pinned at the pole). Moving the
 * vertices on the CPU, not in a shader, keeps the ink pre-pass (an override material) in step.
 * @param list [{base: [x, y, z] scene ft (top of the pole), yaw, len, h, color, pennant}]
 */
function wavingFlags(list) {
  const SEG = 8;
  const pos = [], col = [], idx = [], base = [], along = [], normal = [], phase = [];
  list.forEach((f, k) => {
    const dir = [Math.cos(f.yaw), 0, -Math.sin(f.yaw)];
    const nrm = [Math.sin(f.yaw), 0, Math.cos(f.yaw)];
    const v0 = pos.length / 3;
    for (let i = 0; i <= SEG; i++) {
      const u = i / SEG;
      const half = f.pennant ? (f.h / 2) * (1 - u) : f.h / 2; // pennants taper to a point
      for (const side of [1, -1]) {
        const p = [f.base[0] + dir[0] * f.len * u, f.base[1] - f.h / 2 + side * half, f.base[2] + dir[2] * f.len * u];
        pos.push(...p); base.push(...p); along.push(u * f.len); normal.push(...nrm); phase.push(k * 1.7);
        col.push(...shadeOf(f.color));
      }
    }
    for (let i = 0; i < SEG; i++) {
      const a = v0 + 2 * i;
      idx.push(a, a + 1, a + 3, a, a + 3, a + 2);
    }
  });
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
  g.setAttribute("color", new THREE.Float32BufferAttribute(col, 3));
  g.setIndex(idx);
  g.computeVertexNormals();
  const mesh = new THREE.Mesh(g, makeToonMaterial(0xffffff, { vertexColors: true, side: THREE.DoubleSide, surface: "none" }));
  mesh.frustumCulled = false; // vertices move; the bounding sphere would go stale
  mesh.userData.tick = (t) => {
    const p = g.attributes.position;
    for (let i = 0; i < p.count; i++) {
      const s = along[i];
      const w = Math.sin(s * 0.55 - t * 5 + phase[i]) * 0.09 * s; // amplitude grows toward the tip
      p.setXYZ(i, base[3 * i] + normal[3 * i] * w, base[3 * i + 1] + Math.sin(s * 0.3 - t * 3 + phase[i]) * 0.03 * s, base[3 * i + 2] + normal[3 * i + 2] * w);
    }
    p.needsUpdate = true;
    g.computeVertexNormals();
  };
  return mesh;
}

/** A flagpole (steel) from the ground at plan [x, y] (height h ft, base y0) plus its flag entry. */
function flagpole(p, h, y0, flag, poles, flags) {
  poles.push({ p, y: y0, s: [1, h, 1] });
  flags.push({ base: [p[0], y0 + h - 0.3, -p[1]], ...flag });
}

// ---------- gate building ----------

/** A clock face texture for wall-clock time `date` (hands only: the time in the viewer's zone). */
function drawClock(ctx, date) {
  const S = 256;
  ctx.clearRect(0, 0, S, S);
  ctx.fillStyle = "#f4ead2"; ctx.beginPath(); ctx.arc(128, 128, 120, 0, 7); ctx.fill();
  ctx.lineWidth = 12; ctx.strokeStyle = "#2a4a6b"; ctx.stroke();
  ctx.fillStyle = "#2a4a6b";
  for (let i = 0; i < 12; i++) {
    const a = (i / 12) * 2 * Math.PI;
    ctx.fillRect(128 + Math.sin(a) * 96 - 5, 128 - Math.cos(a) * 96 - 5, 10, 10);
  }
  const hand = (a, len, w) => {
    ctx.lineWidth = w; ctx.lineCap = "round"; ctx.beginPath(); ctx.moveTo(128, 128);
    ctx.lineTo(128 + Math.sin(a) * len, 128 - Math.cos(a) * len); ctx.stroke();
  };
  const m = date.getMinutes(), h = date.getHours() % 12 + m / 60;
  hand((h / 12) * 2 * Math.PI, 60, 12);
  hand((m / 60) * 2 * Math.PI, 90, 8);
}

/**
 * The gate: a coral "brick" hall behind home with a cream clock tower, three glowing arches, the
 * park's name, windows and flags; the cricket park's red pavilion, as the stadium's landmark.
 * Local frame: x along the front, z out of the front (front at z = +d/2).
 */
function gate({ p, yaw }, poles, flags) {
  const g = frameAt(p, [Math.cos(yaw), Math.sin(yaw)]);
  const W = 84, D = 36, H = 40;
  const brick = toon(PALETTE.coral);
  const cream = toon(PALETTE.cream);
  const navy = toon(PALETTE.navy);
  // With yaw 0, frameAt's local z is scene +z = Statcast -y: the front faces away from the field,
  // toward the path and plaza where visitors (and the wide and intro cameras) arrive.
  const hall = new THREE.Group();
  g.add(hall);
  hall.add(box(brick, W, H, D, 0, 0, 0));
  hall.add(box(cream, W + 2, 2.5, D + 2, 0, H, 0)); // cornice
  hall.add(box(cream, W + 1, 4, D + 1, 0, 0, 0)); // plinth
  for (const x of [-W / 2, W / 2]) hall.add(box(cream, 4, H, 4, x, 0, D / 2 - 1.5), box(cream, 4, H, 4, x, 0, -D / 2 + 1.5)); // quoins
  // Gable roof: a triangular prism along x.
  const roof = new THREE.Mesh(new THREE.CylinderGeometry(D / 2 + 2, D / 2 + 2, W + 3, 3).rotateZ(Math.PI / 2).rotateX(-Math.PI / 2).scale(1, 0.62, 1), navy);
  roof.position.set(0, H + 2.5 + 0.31 * (D / 2 + 2), 0); // apex up; its base (at -r/2) on the cornice
  roof.castShadow = true;
  hall.add(roof);
  // Three arches and a sign.
  const archGeo = archGeometry(10, 12, 1.2);
  for (const x of [-22, 0, 22]) {
    const a = new THREE.Mesh(archGeo, windowLitMaterial);
    a.position.set(x, 4, D / 2 + 0.3);
    hall.add(a);
  }
  const sign = new THREE.Mesh(new THREE.PlaneGeometry(56, 6),
    makeToonMaterial(0xffffff, { surface: "none", map: wordmarkTexture(1024, 110) }));
  sign.position.set(0, 25, D / 2 + 0.2);
  hall.add(sign);
  // Upper windows.
  const R = rng(17);
  for (let x = -36; x <= 36; x += 8) {
    if (Math.abs(x) < 12) continue; // the tower covers the middle
    const w = new THREE.Mesh(archGeometry(4, 5, 1), R() < 0.7 ? windowLitMaterial : windowMaterial);
    w.position.set(x, 30, D / 2 + 0.1);
    hall.add(w);
  }
  // Clock tower, standing forward of the hall's front.
  const T = 18, TH = 80;
  hall.add(box(cream, T, TH, T, 0, 0, D / 2 + 4));
  hall.add(box(brick, T + 1.5, 3, T + 1.5, 0, TH - 12, D / 2 + 4));
  hall.add(box(brick, T + 2, 2, T + 2, 0, TH, D / 2 + 4));
  const spire = new THREE.Mesh(new THREE.ConeGeometry(T * 0.78, 26, 4).rotateY(Math.PI / 4).translate(0, TH + 2 + 13, D / 2 + 4), navy);
  spire.castShadow = true;
  hall.add(spire);
  const c = document.createElement("canvas");
  c.width = c.height = 256;
  const ctx = c.getContext("2d");
  drawClock(ctx, new Date());
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  const face = new THREE.Mesh(new THREE.CircleGeometry(6.5, 32), makeToonMaterial(0xffffff, { map: tex, surface: "none", transparent: true }));
  face.position.set(0, TH - 22, D / 2 + 4 + T / 2 + 0.15);
  hall.add(face);
  let shownMinute = -1;
  face.userData.tick = () => { // the clock shows the viewer's own time; redrawn once a minute
    const d = new Date();
    if (d.getMinutes() === shownMinute) return;
    shownMinute = d.getMinutes();
    drawClock(ctx, d);
    tex.needsUpdate = true;
  };
  // Flags: one on the spire, one on each end of the ridge. Positions in plan feet.
  g.updateMatrixWorld(true);
  const world = (x, y, z) => new THREE.Vector3(x, y, z).applyMatrix4(hall.matrixWorld);
  const top = world(0, TH + 28, D / 2 + 4);
  flagpole([top.x, -top.z], 14, top.y, { yaw: Math.PI / 2 + 0.6, len: 9, h: 5.5, color: PALETTE.mustard }, poles, flags);
  for (const x of [-W / 2 + 4, W / 2 - 4]) {
    const q = world(x, H + 14, 0);
    flagpole([q.x, -q.z], 12, q.y, { yaw: Math.PI / 2 + 0.6, len: 8, h: 5, color: PALETTE.teal }, poles, flags);
  }
  return g;
}

// ---------- the rest ----------

/** Fountain: a round stone basin, water, a centre column and a spray of water (not outlined). */
function fountain(p) {
  const g = new THREE.Group();
  g.position.set(...at(p));
  const stone = toon(PALETTE.concreteShade);
  const basin = new THREE.Mesh(new THREE.CylinderGeometry(13, 14, 2.2, 24, 1, true).translate(0, 1.1, 0), stone);
  const rim = new THREE.Mesh(new THREE.TorusGeometry(13.5, 0.7, 6, 24).rotateX(Math.PI / 2).translate(0, 2.2, 0), toon(PALETTE.concrete));
  const water = new THREE.Mesh(new THREE.CircleGeometry(13, 24).rotateX(-Math.PI / 2).translate(0, 1.6, 0), makeToonMaterial(PALETTE.water, { surface: "none" }));
  const column = new THREE.Mesh(new THREE.CylinderGeometry(1.6, 2.4, 7, 10).translate(0, 3.5, 0), stone);
  const bowl = new THREE.Mesh(new THREE.CylinderGeometry(5, 2, 1.6, 16).translate(0, 7.5, 0), toon(PALETTE.concrete));
  const spray = noOutline(new THREE.Mesh(new THREE.ConeGeometry(4.5, 8, 12, 1, true).translate(0, 12, 0),
    new THREE.MeshBasicMaterial({ color: PALETTE.waterLight, transparent: true, opacity: 0.55, side: THREE.DoubleSide, depthWrite: false })));
  for (const m of [basin, rim, column, bowl]) { m.castShadow = true; m.receiveShadow = true; }
  water.receiveShadow = true;
  g.add(basin, rim, water, column, bowl, spray);
  g.userData.tick = (t) => { spray.scale.set(1, 1 + 0.06 * Math.sin(t * 7), 1); };
  return g;
}

/** Gazebo / bandstand: octagonal deck, eight posts, a coral cone roof with a finial. */
function gazebo(p) {
  const g = new THREE.Group();
  g.position.set(...at(p));
  g.add(new THREE.Mesh(new THREE.CylinderGeometry(16, 17, 2.5, 8).translate(0, 1.25, 0), toon(PALETTE.cream)));
  const posts = [];
  for (let i = 0; i < 8; i++) posts.push({ p: [Math.cos((i / 8) * 2 * Math.PI + Math.PI / 8) * 14.5, Math.sin((i / 8) * 2 * Math.PI + Math.PI / 8) * 14.5], y: 2.5 });
  const post = instances(new THREE.CylinderGeometry(0.5, 0.5, 11, 6).translate(0, 5.5, 0), toon(PALETTE.cream), posts);
  g.add(post);
  const roof = new THREE.Mesh(new THREE.ConeGeometry(19, 9, 8).translate(0, 13.5 + 4.5, 0), toon(PALETTE.coral));
  const finial = new THREE.Mesh(new THREE.SphereGeometry(1, 8, 6).translate(0, 23, 0), toon(PALETTE.mustard));
  const rail = new THREE.Mesh(new THREE.TorusGeometry(14.5, 0.25, 4, 8).rotateX(Math.PI / 2).rotateY(Math.PI / 8).translate(0, 5.5, 0), toon(PALETTE.cream));
  for (const m of [roof, finial, rail, g.children[0]]) { m.castShadow = true; m.receiveShadow = true; }
  g.add(roof, finial, rail);
  return g;
}

/** Striped food tents: four-sided peaked tops on short white walls (instanced, two colourings). */
function tents(sites) {
  const top = new THREE.ConeGeometry(9, 7, 4, 1).rotateY(Math.PI / 4).toNonIndexed().translate(0, 8 + 3.5, 0);
  const p = top.attributes.position;
  const col = [];
  for (let i = 0; i < p.count; i += 3) {
    const k = Math.floor(((Math.atan2(p.getZ(i) + p.getZ(i + 1) + p.getZ(i + 2), p.getX(i) + p.getX(i + 1) + p.getX(i + 2)) + Math.PI) / (2 * Math.PI)) * 8) % 2;
    for (let j = 0; j < 3; j++) col.push(...shadeOf(k ? PALETTE.cream : 0xffffff));
  }
  top.setAttribute("color", new THREE.Float32BufferAttribute(col, 3));
  top.computeVertexNormals();
  const walls = new THREE.BoxGeometry(12.5, 8, 12.5).translate(0, 4, 0);
  return [
    instances(top, makeToonMaterial(0xffffff, { vertexColors: true, surface: "none" }), sites),
    instances(walls, toon(PALETTE.cream), sites.map((s) => ({ ...s, color: undefined }))),
  ];
}

/** Parked cars: a body and a cabin, instanced, coloured from the palette. */
function cars(sites) {
  const body = new THREE.BoxGeometry(14, 3, 6.4).translate(0, 2.2, 0);
  const cabin = new THREE.BoxGeometry(7.5, 2.6, 5.6).translate(-0.8, 4.9, 0);
  const wheels = mergeGeometries([-4.5, 4.5].flatMap((x) => [-3.2, 3.2].map((z) => new THREE.CylinderGeometry(1.2, 1.2, 0.8, 10).rotateX(Math.PI / 2).translate(x, 1.2, z))));
  return [
    instances(body, toon(0xffffff), sites),
    instances(cabin, toon(PALETTE.cream), sites.map((s) => ({ ...s, color: undefined }))),
    instances(wheels, toon(0x2b2b2b, "none"), sites.map((s) => ({ ...s, color: undefined })), { cast: false }),
  ];
}

/** Lamp posts along the paths: a dark post with a lamp head that lights (and blooms) at night. */
function lampPosts(sites) {
  const post = instances(new THREE.CylinderGeometry(0.25, 0.35, 14, 6).translate(0, 7, 0), toon(PALETTE.navy, "none"), sites);
  const arm = instances(new THREE.BoxGeometry(2.2, 0.3, 0.3).translate(0.9, 13.6, 0), toon(PALETTE.navy, "none"), sites);
  const head = instances(new THREE.SphereGeometry(0.9, 10, 8).translate(2, 13.2, 0), lampMaterial, sites, { cast: false });
  return [post, arm, head];
}

/** Park benches (wood seat and back on dark legs). */
function benches(sites) {
  const wood = mergeGeometries([new THREE.BoxGeometry(6, 0.3, 1.6).translate(0, 1.6, 0), new THREE.BoxGeometry(6, 1.2, 0.25).translate(0, 2.6, 0.8)]);
  const legs = mergeGeometries([-2.5, 2.5].map((x) => new THREE.BoxGeometry(0.3, 1.6, 1.6).translate(x, 0.8, 0)));
  return [instances(wood, toon(0xb07a4a, "wood"), sites), instances(legs, toon(PALETTE.navy, "none"), sites)];
}

/** Bushes (clusters of three blobs) and boulders, instanced. */
function bushesAndRocks(bushes, rocks) {
  const blob = (r, x, z) => new THREE.IcosahedronGeometry(r, 0).translate(x, r * 0.8, z);
  const bush = mergeGeometries([blob(1, 0, 0), blob(0.75, 0.9, 0.3), blob(0.7, -0.7, 0.6)]);
  const rock = new THREE.DodecahedronGeometry(1, 0).scale(1.2, 0.75, 1).translate(0, 0.45, 0);
  return [
    instances(bush, toon(0xffffff, "foliage"), bushes),
    instances(rock, toon(0xffffff, "rock"), rocks),
  ];
}

/**
 * Waterfall: a ribbon of water from the stream's mouth down the island's cliff, its stripes
 * scrolling downward (texture offset in update), plus a white lip of foam. Not outlined.
 */
function waterfall({ p, dir }) {
  const c = document.createElement("canvas");
  c.width = 64; c.height = 256;
  const x = c.getContext("2d");
  x.fillStyle = "#4fb7c2"; x.fillRect(0, 0, 64, 256);
  const R = rng(3);
  for (let i = 0; i < 40; i++) {
    x.fillStyle = R() < 0.5 ? "#9fdcd8" : "#e8fbf8";
    x.fillRect(R() * 64, R() * 256, 2 + R() * 4, 20 + R() * 50);
  }
  const tex = new THREE.CanvasTexture(c);
  tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
  tex.repeat.set(2, 3);
  tex.colorSpace = THREE.SRGBColorSpace;
  const fall = new THREE.Mesh(new THREE.PlaneGeometry(14, 260), new THREE.MeshBasicMaterial({ map: tex, transparent: true, opacity: 0.92, side: THREE.DoubleSide }));
  const g = frameAt(p, [-dir[1], dir[0]]); // local x along the edge, z outward
  fall.position.set(0, -130, 2);
  g.add(noOutline(fall));
  const foam = noOutline(new THREE.Mesh(new THREE.CylinderGeometry(9, 9, 2, 12, 1, true).scale(1, 1, 0.5).translate(0, -1, 2),
    new THREE.MeshBasicMaterial({ color: 0xf2fbfa, transparent: true, opacity: 0.8 })));
  g.add(foam);
  g.userData.tick = (t) => { tex.offset.y = t * 0.6; };
  return g;
}

/** Rocks hanging under the island (the cricket clip's floating debris): small faceted cones. */
function hangingRocks(centre, sites) {
  const geo = new THREE.ConeGeometry(1, 2.4, 5, 1).rotateX(Math.PI).toNonIndexed();
  geo.computeVertexNormals();
  return instances(geo, toon(0xffffff, "rock"), sites.map((s) => ({ ...s, p: [centre[0] + s.p[0], centre[1] + s.p[1]] })), { cast: false });
}

/**
 * Build the surroundings from the plan. The gate's flags and poles are pushed onto `flags` and
 * `poles` (shared with props.js) so all of the park's flags are one animated mesh (buildFlags).
 * @returns {THREE.Group}
 */
export function buildSurrounds(plan, flags, poles) {
  const g = new THREE.Group();
  g.add(gate(plan.gate, poles, flags));
  for (const f of plan.fountains) g.add(fountain(f));
  g.add(gazebo(plan.gazebo));
  g.add(...tents(plan.tents), ...cars(plan.cars), ...lampPosts(plan.lamps), ...benches(plan.benches));
  g.add(...bushesAndRocks(plan.bushes, plan.rocks));
  g.add(waterfall(plan.waterfall), hangingRocks(plan.hangingCentre, plan.hanging));
  return g;
}

/** The flags and flagpoles of the whole park (from buildSurrounds and props), animated. */
export function buildFlags(flags, poles) {
  const pole = instances(new THREE.CylinderGeometry(0.18, 0.22, 1, 6).translate(0, 0.5, 0), toon(PALETTE.steelLight, "none"), poles);
  return [pole, wavingFlags(flags)];
}

