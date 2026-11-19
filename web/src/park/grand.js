import * as THREE from "three";
import { mergeGeometries } from "three/examples/jsm/utils/BufferGeometryUtils.js";
import { PALETTE, makeToonMaterial, lampMaterial, windowLitMaterial, noOutline } from "../look.js";
import { place } from "./build.js";
import { wallBays, WALL, GRAND } from "./signature.js";
import { slabGeometry, clockTexture, walkwayFans } from "./wall.js";

/*
 * The grand brick wall (right field): the stadium's showpiece, in the park's stylised low-poly look. It IS the right-field
 * outfield wall, on the same bays as the first-built one (wall.js, ?old=oracle):
 *  - storey 1: solid brick on a limestone plinth; a stone string course; storey 2: an arcade of arched openings (real holes:
 *    you see the Falls through them) with stone jambs, voussoirs and keystones, a sill and a balustrade in each;
 *  - brick piers (pilasters) between the bays with limestone bases and capitals; wrought-iron bracket lamps on them;
 *  - a frieze, a dentil band, a heavy cornice, and a balustraded parapet with a lamp on a pedestal over each pier,
 *    and a pennant in each bay;
 *  - at the middle a clock tower: brick shaft with limestone quoins, a gold-lettered stone plaque, clocks on three faces,
 *    a stone belfry, a verdigris copper roof and spire, a gold ball and a flag;
 *  - ivy on a few piers, fans on the walkway behind the arches, warm light in arches and lamps at night (the shared lamp
 *    and window materials, so the night mode's bloom is reused).
 * Brick is a shader (surface.js brickSurface, world space, faded by pixel footprint), so the slabs carry no texture. Everything
 * repeated is one InstancedMesh per material (bays, piers, ivy, lamps), the tower is a few merged meshes: ~30 draws.
 * Local bay frame as wall.js: x along the wall, y up, z toward the field; the walkway is at -z.
 */

const at = ([x, y], h = 0) => [x, h, -y];
const ni = (geo) => { const g = geo.index ? geo.toNonIndexed() : geo; for (const k of Object.keys(g.attributes)) if (!["position", "normal", "uv"].includes(k)) g.deleteAttribute(k); return g; };
const box = (w, h, d, x, y, z) => ni(new THREE.BoxGeometry(w, h, d).translate(x, y + h / 2, z));
const merge = (parts) => mergeGeometries(parts.map(ni));
const G = GRAND;
const TOP = G.h1 + G.h2; // 50: where the arcade's slab ends and the frieze begins

/** The arch's stone ring: n voussoirs round a half circle (inner radius r0, ring width ringW) with a bigger keystone; arc centre (0, cy). */
export function voussoirs(r0, ringW, depth, cy, n = 9) {
  const parts = [];
  const gap = 0.03;
  for (let k = 0; k < n; k++) {
    const key = k === (n - 1) / 2, rOut = r0 + ringW + (key ? 1.0 : 0);
    const a0 = (Math.PI * k) / n + gap, a1 = (Math.PI * (k + 1)) / n - gap;
    const s = new THREE.Shape();
    s.moveTo(r0 * Math.cos(a0), r0 * Math.sin(a0));
    s.lineTo(rOut * Math.cos(a0), rOut * Math.sin(a0));
    s.absarc(0, 0, rOut, a0, a1, false);
    s.lineTo(r0 * Math.cos(a1), r0 * Math.sin(a1));
    s.absarc(0, 0, r0, a1, a0, true);
    const d = depth + (key ? 0.7 : 0);
    parts.push(new THREE.ExtrudeGeometry(s, { depth: d, bevelEnabled: false, curveSegments: 3 }).translate(0, cy, -depth / 2 - (key ? 0.35 : 0)));
  }
  return parts;
}

/** A round-topped plane (w wide, h tall to the crown), for belfry and lantern windows. */
function archPlane(w, h) {
  const r = w / 2, s = new THREE.Shape();
  s.moveTo(-r, 0); s.lineTo(r, 0); s.lineTo(r, h - r); s.absarc(0, h - r, r, 0, Math.PI, false); s.lineTo(-r, 0);
  return new THREE.ShapeGeometry(s, 6);
}

/** One bay's limestone, bay-local. withArch false: no arch parts (the return wall). */
function stoneBay(withArch) {
  const { bay, thick } = WALL;
  const r0 = G.openW / 2 - 0.15, ring = 2.2, depth = thick + 1.0;
  const parts = [
    box(bay + 0.4, G.plinth, thick + 2.2, 0, 0, 0), // plinth
    box(bay + 0.6, 1.6, thick + 1.6, 0, G.h1 - 0.8, 0), // string course
    box(bay + 0.6, 2.5, thick + 1.2, 0, TOP - 2.5, 0), // frieze band
    box(bay + 0.6, 2.0, thick + 4.8, 0, G.cornice, 0), // cornice
    box(bay + 0.4, 1.0, 1.6, 0, G.cornice + 2.0, 0), // parapet base rail
    box(bay + 0.4, 1.0, 1.9, 0, G.parapetTop - 1.0, 0), // parapet top rail
  ];
  for (let k = 0; k < 13; k++) parts.push(box(0.8, 1.1, 0.9, -bay / 2 + 0.9 + k * 1.69, TOP, thick / 2 + 1.05)); // the dentils under the cornice (the frieze face is thick/2 + 0.6; they stand 0.9 proud of it)
  for (let k = 0; k < 9; k++) parts.push(box(0.6, G.parapetTop - 1.0 - (G.cornice + 3.0), 0.6, -bay / 2 + 2.2 + (k * (bay - 4.4)) / 8, G.cornice + 3.0, 0)); // parapet balusters
  if (withArch) {
    const y0 = G.h1 + G.sill, yS = G.h1 + G.spring;
    parts.push(...voussoirs(r0, ring, depth, yS));
    for (const s of [-1, 1]) {
      parts.push(box(ring, G.spring - G.sill, depth, s * (r0 + ring / 2), y0, 0)); // jambs
      parts.push(box(bay / 2 - r0 - ring + 0.2, 1.0, thick + 1.8, s * ((bay / 2 + r0 + ring) / 2 - 0.1), yS - 0.5, 0)); // impost
    }
    parts.push(box(2 * (r0 + ring), 1.1, thick + 1.8, 0, y0 - 1.1, 0)); // sill
    // The balustrade across the opening, seen from the field: rail, seven balusters, handrail.
    parts.push(box(2 * r0 - 0.2, 0.6, 1.0, 0, y0, 0.2), box(2 * r0 - 0.2, 0.8, 1.2, 0, y0 + 3.6, 0.2));
    for (let k = 0; k < 7; k++) parts.push(box(0.6, 3.0, 0.6, -6 + k * 2, y0 + 0.6, 0.2));
  }
  return merge(parts);
}

/** Wrought-iron parts of a bay: the walkway deck and its back rail (storey 2 stands on it), hanging lantern chains. */
function ironBay() {
  const { bay, thick } = WALL;
  const zb = -thick / 2 - G.deck / 2;
  const parts = [
    box(bay, 2.5, G.deck, 0, G.h1 + 2.5, zb), // walkway deck, top at h1 + 5
    box(bay, 0.35, 0.35, 0, G.h1 + 9.4, -thick / 2 - G.deck + 0.2), // back rail
    box(0.35, 4.4, 0.35, -bay / 2 + 0.3, G.h1 + 5, -thick / 2 - G.deck + 0.2), box(0.35, 4.4, 0.35, bay / 2 - 0.3, G.h1 + 5, -thick / 2 - G.deck + 0.2),
  ];
  for (const s of [-1, 1]) parts.push(box(0.14, 4.0, 0.14, s * 3.2, G.h1 + G.spring + 2.2, -thick / 2 - 1.0)); // lantern chains inside the arch
  return merge(parts);
}

/** One pier, centred at x = 0: brick pilaster with stone base and capital, the parapet pedestal and its lamp post. */
function pierBrick() { return merge([box(4.4, 45.3 - G.plinth, WALL.thick + 2.2, 0, G.plinth, 0)]); }
function pierStone() {
  const t = WALL.thick;
  return merge([
    box(6.2, 1.6, t + 3.0, 0, G.plinth, 0), // base
    box(6.4, 2.2, t + 3.2, 0, 45.3, 0), // capital
    box(3.0, G.pedestalTop - 0.8 - (G.cornice + 2.0), 2.6, 0, G.cornice + 2.0, 0), // pedestal
    box(3.8, 0.8, 3.4, 0, G.pedestalTop - 0.8, 0), // its cap
  ]);
}
function pierIron() {
  const t = WALL.thick;
  const parts = [box(0.4, 4.0, 0.4, 0, G.pedestalTop, 0), box(1.9, 0.35, 1.9, 0, G.pedestalTop + 4.0, 0)]; // lamp post and lantern cap
  parts.push(box(0.3, 0.3, 2.4, 0, 14.2, t / 2 + 1.1 + 1.2), box(0.3, 3.4, 0.3, 0, 11.2, t / 2 + 1.1 + 2.4)); // bracket lamp: arm out of the pilaster, hanger
  return merge(parts);
}

// ---------- the clock tower ----------

/** The stone plaque (cream limestone, gold lettering, a carved border). */
function plaqueTexture() {
  const c = document.createElement("canvas");
  c.width = 1024; c.height = 300;
  const g = c.getContext("2d");
  g.fillStyle = "#e9dfc8"; g.fillRect(0, 0, 1024, 300);
  g.strokeStyle = "#a99673"; g.lineWidth = 8; g.strokeRect(14, 14, 996, 272);
  g.strokeStyle = "#c9b790"; g.lineWidth = 3; g.strokeRect(30, 30, 964, 240);
  g.textAlign = "center"; g.textBaseline = "middle";
  g.font = "bold 128px Georgia, 'Times New Roman', serif";
  g.fillStyle = "#6b5120"; g.fillText("PITCH PARK", 516, 158);
  g.fillStyle = "#e3b23c"; g.fillText("PITCH PARK", 510, 150);
  g.fillStyle = "#f6dc8a"; g.globalAlpha = 0.5; g.fillText("PITCH PARK", 507, 146);
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  t.anisotropy = 8;
  return t;
}

function clockTower(brick, stone) {
  const g = new THREE.Group();
  const W = 26, D = 16, cz = 1.5 + 5 - D / 2; // the front face stands 5 ft proud of the wall's
  const fz = cz + D / 2; // front face z
  const copper = makeToonMaterial(0x4fb09a, { surface: "none" }), gold = makeToonMaterial(0xe0b13a, { surface: "none" });
  const st = [], cu = [], au = [], glass = [];
  const shaftH = 74;
  const shaft = new THREE.Mesh(slabGeometry(W, shaftH, D), brick);
  shaft.position.z = cz;
  shaft.castShadow = shaft.receiveShadow = true;
  g.add(shaft);

  st.push(box(W + 2.4, 3.5, D + 2.4, 0, 0, cz)); // plinth
  for (const y of [46, 72]) st.push(box(W + 1.8, 1.6, D + 1.8, 0, y, cz)); // string courses
  // Quoins: alternating long and short stones up all four corners.
  for (let k = 0, y = 4; y < 71; k++, y += 3.0) for (const sx of [-1, 1]) for (const sz of [-1, 1]) {
    const w = (k + (sx > 0) + (sz > 0)) % 2 ? 2.8 : 1.8, d = 4.6 - w;
    st.push(box(w, 1.4, d, sx * (W / 2 - w / 2 + 0.3), y, cz + sz * (D / 2 - d / 2 + 0.3)));
  }
  // The plaque with its frame, and the tall arched window over it (framed by voussoirs and jambs).
  st.push(box(22.4, 8.2, 0.8, 0, 11.4, fz + 0.1));
  st.push(...voussoirs(4.5, 1.8, 1.6, 32, 7).map((geo) => geo.translate(0, 0, fz - 0.2)));
  for (const s of [-1, 1]) st.push(box(1.8, 10, 1.6, s * 5.4, 22, fz - 0.2));
  st.push(box(14, 1.1, 2.4, 0, 20.9, fz)); // sill
  const arch = archPlane(9, 14.5); arch.translate(0, 22, fz + 0.45); glass.push(arch);
  // The belfry stage: stone, an arched opening on each face.
  st.push(box(W - 1.6, 16.4, D - 1.6, 0, 73.6, cz));
  // Face distances from the shaft's axis (z = cz): front (D - 1.6) / 2, sides (W - 1.6) / 2; geometry faces +z, then turned to its face.
  const onFace = (geo, rot, dist) => geo.rotateY(rot).translate(Math.sin(rot) * dist, 0, Math.cos(rot) * dist + cz);
  for (const [rot, dist] of [[0, (D - 1.6) / 2], [Math.PI / 2, (W - 1.6) / 2], [-Math.PI / 2, (W - 1.6) / 2]]) {
    glass.push(onFace(archPlane(6.4, 11).translate(0, 76, 0), rot, dist + 0.2));
    for (const r of voussoirs(3.2, 1.3, 0.9, 76 + 11 - 3.2, 7)) st.push(onFace(r, rot, dist + 0.1));
  }
  // The cornice, a dentil band, the roof's corner pinnacles.
  st.push(box(W + 4, 2.2, D + 4, 0, 90, cz));
  for (let k = 0; k < 14; k++) st.push(box(0.8, 1.1, 0.9, -W / 2 - 0.7 + k * 1.98, 89, fz + 0.9));
  for (const sx of [-1, 1]) for (const sz of [-1, 1]) {
    const x = sx * (W / 2 + 0.8), z = cz + sz * (D / 2 + 0.8);
    st.push(box(2.6, 2.6, 2.6, x, 92.2, z));
    cu.push(new THREE.ConeGeometry(1.7, 8, 6).translate(x, 94.8 + 4, z));
  }
  // The roof: a steep copper hip, a stone lantern with lit arches, a second roof and the spire with a gold ball.
  cu.push(new THREE.CylinderGeometry(5.2, (W + 3) * 0.7, 17, 4).rotateY(Math.PI / 4).translate(0, 92.2 + 8.5, cz));
  st.push(box(9.4, 9.6, 9.4, 0, 109.2, cz), box(11.4, 1.3, 11.4, 0, 118.8, cz), box(11.4, 1.3, 11.4, 0, 108.4, cz));
  for (const [dx, dz, ry] of [[0, 4.8, 0], [0, -4.8, Math.PI], [4.8, 0, Math.PI / 2], [-4.8, 0, -Math.PI / 2]]) {
    glass.push(archPlane(3.6, 7).rotateY(ry).translate(dx, 111, cz + dz));
  }
  cu.push(new THREE.ConeGeometry(8.3, 15, 4).rotateY(Math.PI / 4).translate(0, 120.1 + 7.5, cz));
  cu.push(new THREE.ConeGeometry(2.6, 14, 8).translate(0, 135.1 + 7, cz));
  au.push(new THREE.IcosahedronGeometry(1.9, 1).translate(0, 149.7, cz), new THREE.CylinderGeometry(0.25, 0.25, 4, 5).translate(0, 152, cz));
  const finialY = 151;

  const mesh = (parts, mat, shadow = true) => { const m = new THREE.Mesh(merge(parts), mat); m.castShadow = shadow; m.receiveShadow = true; g.add(m); return m; };
  mesh(st, stone); mesh(cu, copper); mesh(au, gold);
  g.add(noOutline(new THREE.Mesh(merge(glass), windowLitMaterial)));

  // Clocks on the front and both sides: a stone surround, the dial (unlit, as in wall.js), at y = 60.
  const dial = new THREE.CircleGeometry(8, 32), ring = new THREE.CylinderGeometry(9.6, 9.6, 0.9, 20).rotateX(Math.PI / 2);
  const dialMat = new THREE.MeshBasicMaterial({ map: clockTexture() });
  for (const [ry, dist] of [[0, D / 2], [Math.PI / 2, W / 2], [-Math.PI / 2, W / 2]]) {
    const f = new THREE.Group();
    f.position.set(0, 60, cz);
    f.rotation.y = ry;
    const r = new THREE.Mesh(ring, stone); r.position.z = dist + 0.2; r.castShadow = true; f.add(r);
    const d = noOutline(new THREE.Mesh(dial, dialMat)); d.position.z = dist + 0.75; f.add(d);
    g.add(f);
  }
  // The plaque: gold letters on stone, with a little glow at night (emissive map, like the scoreboard).
  const ptex = plaqueTexture();
  const plaque = noOutline(new THREE.Mesh(new THREE.PlaneGeometry(20.8, 6.1), makeToonMaterial(0xffffff, { map: ptex, emissiveMap: ptex, emissive: 0xffffff, emissiveIntensity: 0.25, surface: "none", polygonOffset: true, polygonOffsetFactor: -2, polygonOffsetUnits: -8 })));
  plaque.position.set(0, 11.4 + 4.1, fz + 0.62);
  plaque.position.y = 11.4 + 4.1;
  g.add(plaque);
  return { g, cz, finialY, flagTop: finialY + 14 };
}

// ---------- ivy ----------

function ivy(sites, wallBay) {
  const geo = new THREE.IcosahedronGeometry(1, 1).scale(1, 1.15, 0.55);
  const m = new THREE.InstancedMesh(geo, makeToonMaterial(0xffffff, { surface: "foliage" }), sites.length);
  const c = new THREE.Color();
  const greens = [0x3f8f4a, 0x2f7a45, 0x5ea24c, 0x2c6b4c];
  sites.forEach((s, i) => {
    place(m, i, at(wallBay(s.b, s.lx, s.lz), s.y), s.b.yaw, s.s);
    m.setColorAt(i, c.set(greens[i % greens.length]));
  });
  m.castShadow = true;
  return m;
}

export function buildGrandWall(flags, poles) {
  const g = new THREE.Group();
  const bays = wallBays();
  const n = bays.length, main = bays.filter((b) => !b.ret), m = main.length;
  const { bay, thick } = WALL;
  const brickMat = makeToonMaterial(0xb4503c, { surface: "brick", side: THREE.DoubleSide });
  const stoneMat = makeToonMaterial(0xe8dec6, { surface: "plaster" });
  const ironMat = makeToonMaterial(0x2a2f33, { surface: "none" });
  const mk = (geo, mat, count, shadow = true) => { const i = new THREE.InstancedMesh(geo, mat, count); i.castShadow = shadow; i.receiveShadow = true; g.add(i); return i; };

  const lower = mk(slabGeometry(bay, G.h1, thick), brickMat, n);
  const arcade = mk(slabGeometry(bay, G.h2, thick, G.openW, G.spring, G.sill), brickMat, m);
  const solid = mk(slabGeometry(bay, G.h2, thick), brickMat, 1); // the return wall's upper storey (no arches)
  const stoneA = mk(stoneBay(true), stoneMat, m), stoneR = mk(stoneBay(false), stoneMat, 1);
  const iron = mk(ironBay(), ironMat, m);
  const lampsArch = new THREE.InstancedMesh(new THREE.BoxGeometry(1.0, 1.6, 1.0), lampMaterial, 2 * m);
  const dir = (b) => [Math.cos(b.yaw), Math.sin(b.yaw)], zd = (b) => [Math.sin(b.yaw), -Math.cos(b.yaw)];
  const wallBay = (b, lx, lz) => [b.p[0] + dir(b)[0] * lx + zd(b)[0] * lz, b.p[1] + dir(b)[1] * lx + zd(b)[1] * lz];

  const colours = [PALETTE.coral, PALETTE.mustard, PALETTE.teal, PALETTE.cream];
  let ret = null;
  bays.forEach((b, i) => {
    const sx = [b.len / bay, 1, 1];
    place(lower, i, at(b.p), b.yaw, sx);
    if (b.ret) { place(solid, 0, at(b.p, G.h1), b.yaw, sx); place(stoneR, 0, at(b.p), b.yaw, sx); ret = b; return; }
    place(arcade, i, at(b.p, G.h1), b.yaw, sx);
    place(stoneA, i, at(b.p), b.yaw, sx);
    place(iron, i, at(b.p), b.yaw, sx);
    for (const [k, s] of [-1, 1].entries()) place(lampsArch, 2 * i + k, at(wallBay(b, s * 3.2 * sx[0], -thick / 2 - 1.0), G.h1 + G.spring + 1.2), b.yaw, 1);
    // A pennant on a staff in the middle of each bay's parapet.
    poles.push({ p: b.p, y: G.parapetTop, s: [1, 14, 1] });
    flags.push({ base: [b.p[0], G.parapetTop + 14 - 0.3, -b.p[1]], yaw: 0, len: 12, h: 4.6, color: colours[i % 4], pennant: true });
  });
  g.add(noOutline(lampsArch));

  // Piers: at the start of the first bay and the end of each main bay (yaw = the bisector of the two walls it joins); the last one
  // is the corner, wider, where the wall turns into foul ground.
  const piers = [];
  const yawOf = (a, b) => Math.atan2(Math.sin(a) + Math.sin(b), Math.cos(a) + Math.cos(b));
  const endOf = (b, s) => [b.p[0] + dir(b)[0] * (b.len / 2) * s, b.p[1] + dir(b)[1] * (b.len / 2) * s];
  piers.push({ p: endOf(main[0], -1), yaw: main[0].yaw, k: 1.6 }); // the join with the return wall
  main.forEach((b, i) => piers.push({ p: endOf(b, 1), yaw: i + 1 < m ? yawOf(b.yaw, main[i + 1].yaw) : b.yaw, k: i + 1 < m ? 1 : 1.6 }));
  const pb = mk(pierBrick(), brickMat, piers.length), ps = mk(pierStone(), stoneMat, piers.length), pi = mk(pierIron(), ironMat, piers.length);
  const lampsPier = new THREE.InstancedMesh(new THREE.BoxGeometry(1.7, 2.6, 1.7), lampMaterial, 2 * piers.length);
  piers.forEach((q, i) => {
    for (const mesh of [pb, ps, pi]) place(mesh, i, at(q.p), q.yaw, [q.k, 1, q.k]);
    place(lampsPier, 2 * i, at(q.p, G.pedestalTop + 1.3), q.yaw, 1);
    place(lampsPier, 2 * i + 1, at(wallBay({ p: q.p, yaw: q.yaw }, 0, thick / 2 + 3.5), 10.0), q.yaw, [0.6, 0.9, 0.6]);
  });
  g.add(noOutline(lampsPier));

  // The clock tower at the middle bay.
  const mb = main[Math.floor(main.length / 2)];
  const t = clockTower(brickMat, stoneMat);
  t.g.position.set(...at(mb.p));
  t.g.rotation.y = mb.yaw;
  g.add(t.g);
  const fp = wallBay(mb, 0, t.cz);
  poles.push({ p: fp, y: t.finialY, s: [1, 14, 1] });
  flags.push({ base: [fp[0], t.flagTop - 0.3, -fp[1]], yaw: 0, len: 20, h: 8, color: PALETTE.coral, pennant: true });

  // Ivy on the foot of a few piers (and the tower's corner), as blobs pressed against the brick.
  const sites = [];
  piers.forEach((q, i) => {
    if (i % 3 !== 1) return;
    const b = { p: q.p, yaw: q.yaw };
    for (let k = 0; k < 7; k++) sites.push({ b, lx: ((k * 7) % 5 - 2) * 0.55, lz: thick / 2 + 1.5, y: 3.4 + k * 2.3 + (k % 2) * 0.8, s: [1.4 + ((k * 3) % 4) * 0.3, 1.5, 1.4] });
    for (let k = 0; k < 3; k++) sites.push({ b, lx: (k - 1) * 2.4, lz: thick / 2 + 2.4, y: 2.4, s: [1.9, 1.3, 1.9] });
  });
  for (let k = 0; k < 9; k++) sites.push({ b: mb, lx: -13.2 + (k % 2) * 0.4, lz: t.cz + 8.4 - (k % 3) * 0.4, y: 4 + k * 2.6, s: [1.5, 1.7, 1.5] });
  g.add(ivy(sites, wallBay));

  // Fans on the walkway, seen through the arches over the balustrades.
  g.add(...walkwayFans(main, mb, G.h1 + 5.2));
  return g;
}
