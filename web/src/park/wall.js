import * as THREE from "three";
import { mergeGeometries } from "three/examples/jsm/utils/BufferGeometryUtils.js";
import { PALETTE, makeToonMaterial, lampMaterial, noOutline } from "../look.js";
import { place } from "./build.js";
import { isOld } from "./old.js";
import { wordmarkTexture } from "./brand.js";
import { wallBays, WALL } from "./signature.js";
import { rng } from "./layout.js";
import { HOME_SHIRTS, AWAY_SHIRTS, NEUTRAL_SHIRTS } from "./crowd.js";

/*
 * The right-field wall, as at Oracle Park but in our own architecture: it IS the outfield wall from the right-field
 * bleachers to the foul pole, continuing round the corner into the stands, with the padded fence (stands.js) at its foot.
 *  - lower storey: solid brick, 26 ft, plinth and stringcourse in the stands' dark-green trim;
 *  - upper storey: a two-storey-feel arcade of tall arches (real holes: you see the Falls through them), a balustrade
 *    at the sill, a walkway behind with a back rail and standing fans;
 *  - a cornice and cap in the stands' dark greens, a lantern on every pier (lit at night), pennant poles;
 *  - at the middle a brick clock tower with the PITCH PARK lettering band, a pyramid roof and a finial pennant.
 * Every repeated part is one InstancedMesh (lower slabs, arcade slabs, two trims, lanterns, fans): ~12 draws in all.
 * Local bay frame: x along the wall, y up, z toward home (the field side); the walkway is at -z.
 */

const GREEN = isOld("structure") ? { mid: PALETTE.cream, dark: PALETTE.coral } : { mid: 0x3b7a63, dark: 0x1f4538 }; // as stands.js
const at = ([x, y], h = 0) => [x, h, -y];

/** Brick coursing: 4 courses per 8 ft tile, bricks 4 ft long, offset on alternate courses (big, so it does not shimmer at 350 ft). */
function brickTexture() {
  const c = document.createElement("canvas");
  c.width = c.height = 128;
  const g = c.getContext("2d");
  g.fillStyle = "#c9ac90"; g.fillRect(0, 0, 128, 128); // mortar
  const tones = ["#b4503c", "#a64836", "#bc5a42", "#af4c39"];
  for (let row = 0; row < 4; row++) for (let i = -1; i < 3; i++) {
    g.fillStyle = tones[(row * 3 + i + 5) % 4];
    g.fillRect(i * 64 + (row % 2) * 32 + 2, row * 32 + 2, 60, 28);
  }
  const t = new THREE.CanvasTexture(c);
  t.wrapS = t.wrapT = THREE.RepeatWrapping;
  t.repeat.set(1 / 8, 1 / 8);
  t.colorSpace = THREE.SRGBColorSpace;
  t.anisotropy = 8;
  return t;
}

/** A slab bay w x h x thick (UVs in ft, so the brick tile repeats every 8 ft); with openW, a round-topped hole from `sill` up to spring + radius. */
export function slabGeometry(w, h, thick, openW = 0, spring = 0, sill = 0) {
  const s = new THREE.Shape();
  s.moveTo(-w / 2, 0); s.lineTo(w / 2, 0); s.lineTo(w / 2, h); s.lineTo(-w / 2, h); s.closePath();
  if (openW) {
    const r = openW / 2, hole = new THREE.Path();
    hole.moveTo(-r, sill); hole.lineTo(r, sill); hole.lineTo(r, spring);
    hole.absarc(0, spring, r, 0, Math.PI, false);
    hole.lineTo(-r, sill);
    s.holes.push(hole);
  }
  return new THREE.ExtrudeGeometry(s, { depth: thick, bevelEnabled: false, curveSegments: 8 }).translate(0, 0, -thick / 2);
}

const box = (w, h, d, x, y, z) => new THREE.BoxGeometry(w, h, d).translate(x, y + h / 2, z).toNonIndexed();
const merge = (parts) => mergeGeometries(parts);

/** Mid-green trim of one bay: plinth, stringcourse, arch jambs, the cornice body. */
function trimMid({ bay, h1, h2, thick, openW, spring }) {
  const top = h1 + h2, r = openW / 2;
  return merge([
    box(bay + 0.4, 2.6, thick + 1.4, 0, 0, 0), // plinth
    box(bay + 0.5, 1.6, thick + 1.6, 0, h1 - 1.6, 0), // stringcourse under the walkway
    box(1.2, spring - 3.5, thick + 0.9, -r - 0.7, h1 + 3.5, 0), box(1.2, spring - 3.5, thick + 0.9, r + 0.7, h1 + 3.5, 0), // jambs
    box(openW + 2.8, 1.1, thick + 0.9, 0, h1 + spring + r + 0.9, 0), // keystone course over the arch
    box(bay + 0.6, 2.2, thick + 2.2, 0, top, 0), // cornice body
  ]);
}

/** Dark-green trim: cornice cap, walkway deck, back rail, the pier lantern posts (at the bay's far end). */
function trimDark({ bay, h1, h2, thick, deck }) {
  const top = h1 + h2;
  return merge([
    box(bay + 1, 1.0, thick + 2.8, 0, top + 2.2, 0), // cornice cap
    box(bay, 2.5, deck, 0, h1 - 1, -thick / 2 - deck / 2), // walkway deck (top 1.5 ft above the arcade's base, so standing fans show over the balustrade)
    box(bay, 0.35, 0.35, 0, h1 + 4.9, -thick / 2 - deck + 0.2), // back rail
    box(0.35, 3.4, 0.35, -bay / 2 + 0.3, h1 + 1.5, -thick / 2 - deck + 0.2), box(0.35, 3.4, 0.35, bay / 2 - 0.3, h1 + 1.5, -thick / 2 - deck + 0.2), // its posts
    box(1.4, 6.5, 1.4, bay / 2, top + 3.2, thick / 2 - 0.2), // lantern post on the pier
  ]);
}

/** A clock face texture: cream dial, tick marks, hands at the load-time clock. */
function clockTexture() {
  const c = document.createElement("canvas");
  c.width = c.height = 256;
  const g = c.getContext("2d");
  g.fillStyle = "#1f4538"; g.beginPath(); g.arc(128, 128, 126, 0, 7); g.fill();
  g.fillStyle = "#f4ead2"; g.beginPath(); g.arc(128, 128, 112, 0, 7); g.fill();
  g.fillStyle = "#1d3550";
  for (let i = 0; i < 12; i++) { const a = (i / 12) * 2 * Math.PI; g.fillRect(128 + Math.sin(a) * 92 - 6, 128 - Math.cos(a) * 92 - 6, 12, 12); }
  const d = new Date(), m = d.getMinutes(), h = (d.getHours() % 12) + m / 60;
  g.strokeStyle = "#1d3550"; g.lineCap = "round";
  for (const [a, len, w] of [[(h / 12) * 2 * Math.PI, 62, 13], [(m / 60) * 2 * Math.PI, 92, 8]]) {
    g.lineWidth = w; g.beginPath(); g.moveTo(128, 128); g.lineTo(128 + Math.sin(a) * len, 128 - Math.cos(a) * len); g.stroke();
  }
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  t.anisotropy = 8;
  return t;
}

/** The clock tower at the middle bay: brick shaft, cornice, a clock, the lettering band, a pyramid roof and a finial pennant. Local frame as a bay. */
function tower(brickMat) {
  const g = new THREE.Group();
  const W = 30, D = 14, H = 66, cz = 4 - D / 2; // front face 4 ft proud of the wall
  const shaft = new THREE.Mesh(slabGeometry(W, H, D), brickMat);
  shaft.position.z = cz;
  g.add(shaft);
  const mid = makeToonMaterial(GREEN.mid, { surface: "none" }), dark = makeToonMaterial(GREEN.dark, { surface: "none" });
  const trim = (m, parts) => { const t = new THREE.Mesh(merge(parts), m); t.castShadow = t.receiveShadow = true; g.add(t); };
  trim(mid, [box(W + 1, 2.6, D + 1.4, 0, 0, cz), box(W + 1.2, 1.6, D + 1.6, 0, 24, cz), box(W + 1.2, 1.6, D + 1.6, 0, 44, cz), box(W + 2, 2.4, D + 2.4, 0, H, cz)]);
  trim(dark, [box(W + 3, 1.0, D + 3.4, 0, H + 2.4, cz)]);
  const roof = new THREE.Mesh(new THREE.ConeGeometry((W + 3) * 0.72, 17, 4).rotateY(Math.PI / 4).translate(0, H + 3.4 + 8.5, cz), dark);
  roof.castShadow = true;
  g.add(roof);
  // Clock (cream dial, both the front and, flat, the back is skipped: nothing sees it) and the lettering band.
  const clock = noOutline(new THREE.Mesh(new THREE.CircleGeometry(7, 28), new THREE.MeshBasicMaterial({ map: clockTexture() })));
  clock.position.set(0, 54, cz + D / 2 + 0.25);
  g.add(clock);
  const band = noOutline(new THREE.Mesh(new THREE.PlaneGeometry(28, 7), new THREE.MeshBasicMaterial({ map: wordmarkTexture() })));
  band.position.set(0, 14.5, cz + D / 2 + 0.25);
  g.add(band);
  return { g, cz, finialY: H + 3.4 + 17 };
}

/** The wall: returns a Group. flags/poles are the park's shared pennant lists. */
export function buildWall(flags, poles) {
  const g = new THREE.Group();
  const bays = wallBays();
  const n = bays.length;
  const brickMat = makeToonMaterial(0xffffff, { map: brickTexture(), surface: "none", side: THREE.DoubleSide });
  const mk = (geo, mat) => { const m = new THREE.InstancedMesh(geo, mat, n); m.castShadow = m.receiveShadow = true; g.add(m); return m; };
  const lower = mk(slabGeometry(WALL.bay, WALL.h1, WALL.thick), brickMat);
  const arcade = mk(slabGeometry(WALL.bay, WALL.h2, WALL.thick, WALL.openW, WALL.spring, WALL.sill), brickMat);
  const tmid = mk(trimMid(WALL), makeToonMaterial(GREEN.mid, { surface: "none" }));
  const tdark = mk(trimDark(WALL), makeToonMaterial(GREEN.dark, { surface: "none" }));
  const lamps = new THREE.InstancedMesh(new THREE.BoxGeometry(2, 2.6, 2).translate(0, 1.3, 0), lampMaterial, n);
  const top = WALL.h1 + WALL.h2;
  const colours = [PALETTE.coral, PALETTE.mustard, PALETTE.teal, PALETTE.cream];
  bays.forEach((b, i) => {
    const sx = [b.len / WALL.bay, 1, 1];
    place(lower, i, at(b.p), b.yaw, sx); place(arcade, i, at(b.p, WALL.h1), b.yaw, sx);
    place(tmid, i, at(b.p), b.yaw, sx); place(tdark, i, at(b.p), b.yaw, sx);
    const dx = Math.cos(b.yaw), dy = Math.sin(b.yaw); // along the wall, in plan
    place(lamps, i, at([b.p[0] + (dx * b.len) / 2, b.p[1] + (dy * b.len) / 2], top + 9.7), b.yaw, 1);
    if (i % 2 === 0 && !b.ret) { // pennant poles on the cornice, every other bay
      poles.push({ p: b.p, y: top + 3.2, s: [1, 16, 1] });
      flags.push({ base: [b.p[0], top + 3.2 + 16 - 0.3, -b.p[1]], yaw: 0, len: 12, h: 4.6, color: colours[(i / 2) % 4], pennant: true });
    }
  });
  g.add(noOutline(lamps));

  // The corner: a full-height pier where the wall turns from the fence line into foul ground, so it has no open end.
  const corner = bays.length > 1 ? bays[bays.length - 2] : null;
  if (corner) {
    const dx = Math.cos(corner.yaw), dy = Math.sin(corner.yaw);
    const p = [corner.p[0] + (dx * corner.len) / 2, corner.p[1] + (dy * corner.len) / 2];
    const pier = new THREE.Mesh(box(7, top + 3.2, 7, 0, 0, 0), makeToonMaterial(GREEN.dark, { surface: "none" }));
    pier.position.set(...at(p));
    pier.castShadow = pier.receiveShadow = true;
    g.add(pier);
  }

  // The clock tower at the middle bay.
  const main = bays.filter((b) => !b.ret);
  const mb = main[Math.floor(main.length / 2)];
  const t = tower(brickMat);
  t.g.position.set(...at(mb.p));
  t.g.rotation.y = mb.yaw;
  g.add(t.g);
  const zdir = [Math.sin(mb.yaw), -Math.cos(mb.yaw)];
  const fp = [mb.p[0] + zdir[0] * t.cz, mb.p[1] + zdir[1] * t.cz];
  poles.push({ p: fp, y: t.finialY, s: [1, 14, 1] });
  flags.push({ base: [fp[0], t.finialY + 14 - 0.3, -fp[1]], yaw: 0, len: 15, h: 6, color: PALETTE.coral, pennant: true });

  // Fans on the walkway, standing behind the balustrade and seen through the arches (instanced, as crowd.js's).
  const R = rng(5);
  const spots = [];
  for (const b of main) for (const row of [-WALL.thick / 2 - 2.4, -WALL.thick / 2 - 5.4]) for (const x of [-5, -1.7, 1.7, 5]) {
    if (b === mb || R() < 0.35) continue;
    const dx = Math.cos(b.yaw), dy = Math.sin(b.yaw), zx = Math.sin(b.yaw), zy = -Math.cos(b.yaw);
    spots.push({ p: [b.p[0] + dx * (x + (R() - 0.5) * 0.6) + zx * row, b.p[1] + dy * (x + (R() - 0.5) * 0.6) + zy * row], yaw: b.yaw, k: 0.92 + 0.16 * R() });
  }
  const body = new THREE.InstancedMesh(new THREE.BoxGeometry(1.25, 2.9, 0.9).translate(0, 1.45, 0), makeToonMaterial(0xffffff, { surface: "none" }), spots.length);
  const head = new THREE.InstancedMesh(new THREE.IcosahedronGeometry(0.42, 1).translate(0, 3.35, 0), makeToonMaterial(0xffffff, { surface: "none" }), spots.length);
  const skins = [0xf1c7a1, 0xd9a07a, 0xb57a52, 0x8a5a3a, 0x5e3b28];
  const c = new THREE.Color();
  spots.forEach((s, i) => {
    place(body, i, at(s.p, WALL.h1 + 1.5), s.yaw, [1, s.k, 1]);
    place(head, i, at(s.p, WALL.h1 + 1.5 + 3.35 * (s.k - 1)), s.yaw);
    const u = R(), pool = u < 0.68 ? HOME_SHIRTS : u < 0.76 ? AWAY_SHIRTS : NEUTRAL_SHIRTS;
    body.setColorAt(i, c.set(pool[Math.floor(R() * pool.length)]));
    head.setColorAt(i, c.set(skins[Math.floor(R() * skins.length)]));
  });
  body.castShadow = head.castShadow = true;
  g.add(body, head);
  return g;
}
