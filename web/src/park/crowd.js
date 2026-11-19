import * as THREE from "three";
import { mergeGeometries } from "three/examples/jsm/utils/BufferGeometryUtils.js";
import { PALETTE, makeToonMaterial } from "../look.js";
import { rng } from "./layout.js";
import { place } from "./build.js";
import { addHaze } from "./haze.js";

/*
 * The crowd (replaces stands.js's old half-full random-colour crowd; ?old=crowd restores that).
 *
 * Why it read as noise before: every fan's shirt was an independent draw from ten colours. A real
 * crowd reads as people because of STRUCTURE at TV distance: a team-colour majority, sections of
 * same-coloured fans split by aisles, density that follows the sightlines (packed behind home, thin
 * in the far corners), and a few fans standing. planFans() is that structure as pure data (tested);
 * buildCrowd() turns it into instanced meshes (seats, bodies, heads, caps, raised arms, stairs).
 *
 * Idle motion is a vertex-shader bob (a small hop by a per-fan phase; standing fans hop more), one
 * time uniform for the whole crowd, so it costs nothing on the CPU. The outline pre-pass does not see
 * the bob (it draws the still pose): the offsets are well under a pixel at any camera distance
 * where ink lines are drawn, so nothing visibly tears.
 */

// Home team wears the park's coral; the away pockets wear purple (clear of the teal seats and coral roof).
export const HOME_SHIRTS = [0xe5604b, 0xe5604b, 0xc9402f, 0xf4ead2];
export const AWAY_SHIRTS = [0x7b52ab, 0x7b52ab, 0x5d3b8c, 0xb59bd8];
export const NEUTRAL_SHIRTS = [0xf4ead2, 0xf2c14e, 0xffffff, 0x3fa7a0, 0x2a4a6b];
const SKINS = [0xf1c7a1, 0xd9a07a, 0xb57a52, 0x8a5a3a, 0x5e3b28];
const HAIRS = [0x2a1d16, 0x4a3324, 0x8a6a3c, 0xd8b45a, 0x1a1a1a, 0x7d4a2a, 0xb0b0b0];

const smooth = (a, b, x) => { const t = Math.min(1, Math.max(0, (x - a) / (b - a))); return t * t * (3 - 2 * t); };

/** Chance a seat is taken: packed behind home, thinner toward the far corners, bleachers and the back rows. */
export function occupancy(s) {
  const near = 1 - smooth(60, 330, Math.hypot(s.p[0], s.p[1])); // 1 behind home, 0 at the far corners
  const base = s.deck === "bleacher" ? 0.6 : s.deck === "upper" ? 0.5 + 0.38 * near : 0.64 + 0.35 * near;
  return base * (1 - 0.15 * (s.row / Math.max(1, s.rows - 1)));
}

/**
 * Share of a section's fans who are away fans. Visitors sit down the third-base side of the lower
 * deck (a pocket), a few other sections are visitor groups, everything else is home.
 */
export function awayShare(s, sectionRoll) {
  const left = s.p[0] < 0, d = Math.hypot(s.p[0], s.p[1]);
  if (s.deck === "lower") return left && d > 80 && d < 300 ? 0.72 : sectionRoll < 0.07 ? 0.55 : 0.05;
  if (s.deck === "upper") return left && d < 260 ? 0.4 : sectionRoll < 0.05 ? 0.5 : 0.05;
  return left ? 0.25 : sectionRoll < 0.15 ? 0.5 : 0.06;
}

/**
 * Who sits where. seats: [{p: [x, y] ft, h, yaw, deck, row, rows, sec}] (sec = section id, same
 * for every row of one section). Returns one entry per seat: null (empty) or
 * {team: "home"|"away"|"neutral", shirt, skin, hair, cap, standing, armsUp}.
 */
export function planFans(seats, seed = 11) {
  const R = rng(seed);
  const rolls = new Map();
  const roll = (key) => { if (!rolls.has(key)) rolls.set(key, R()); return rolls.get(key); };
  const pick = (a) => a[Math.floor(R() * a.length)];
  return seats.map((s) => {
    if (R() > occupancy(s)) return null;
    const key = `${s.deck}${s.sec}`, r = roll(key);
    const away = R() < awayShare(s, r);
    const neutral = R() < 0.1;
    const team = neutral ? "neutral" : away ? "away" : "home";
    const shirts = neutral ? NEUTRAL_SHIRTS : away ? AWAY_SHIRTS : HOME_SHIRTS;
    const lowFront = s.deck === "lower" && s.row < s.rows - 3;
    // Standing fans: mostly in the lower bowl; some sections are livelier than others.
    const standing = R() < (lowFront ? 0.03 + 0.08 * roll(`loud${key}`) : 0.015);
    const cap = R() < 0.35;
    return {
      team, shirt: pick(shirts), skin: pick(SKINS), standing, armsUp: standing && R() < 0.65,
      hair: cap ? (team === "away" ? 0x5d3b8c : team === "home" ? 0xc9402f : 0x2a4a6b) : pick(HAIRS), cap,
    };
  });
}

/** A toon material whose vertices hop by aBob = (amplitude ft, phase 0..1): a shared clock uniform. */
function bobMaterial(uTime, k) {
  const m = makeToonMaterial(0xffffff, { surface: "none" });
  const park = m.onBeforeCompile;
  m.onBeforeCompile = (s, r) => {
    park(s, r);
    s.uniforms.uTime = uTime;
    s.vertexShader = s.vertexShader
      .replace("#include <common>", "#include <common>\nuniform float uTime;\nattribute vec2 aBob;")
      .replace("#include <begin_vertex>", `#include <begin_vertex>
        transformed.y += aBob.x * ${k.toFixed(2)} * pow(0.5 + 0.5 * sin(uTime * (2.6 + aBob.y * 1.4) + aBob.y * 43.0), 2.0);`);
  };
  addHaze(m); // chains after the bob patch above
  m.customProgramCacheKey = () => `crowdbob${k}`;
  return m;
}

/** A copy of a BoxGeometry without the given faces (0..5). */
function dropFaces(geo, faces) {
  const keep = [...geo.index.array].filter((_, i) => !faces.includes(Math.floor(i / 6)));
  geo.setIndex(keep);
  return geo;
}

const addInstances = (geo, mat, n, bob) => {
  const m = new THREE.InstancedMesh(geo, mat, n);
  m.geometry.setAttribute("aBob", new THREE.InstancedBufferAttribute(bob, 2));
  return m;
};

const Mt = new THREE.Matrix4(), Qt = new THREE.Quaternion(), Qz = new THREE.Quaternion(), Vp = new THREE.Vector3(), Vs = new THREE.Vector3(1, 1, 1);
const YAXIS = new THREE.Vector3(0, 1, 0), ZAXIS = new THREE.Vector3(0, 0, 1);

/**
 * Seats, fans and stairs as instanced meshes.
 * @param seats [{p, h, yaw, color, deck, row, rows, sec}] (stands.js)
 * @param stairs [{p, h, yaw}] one per aisle tread: a lighter step strip
 * @returns Group
 */
export function buildCrowd(seats, stairs, seed = 11) {
  const g = new THREE.Group();
  const R = rng(seed + 1);
  const fans = planFans(seats, seed);
  const people = seats.map((s, i) => [s, fans[i]]).filter(([, f]) => f);
  const c = new THREE.Color();
  const uTime = { value: 0 };

  // Seats: pan and back; the back is on the side away from the field (+z local).
  // Hidden faces dropped (BoxGeometry face order: +x -x +y -y +z -z): pan's bottom and back, the back's bottom. 24 -> 18 triangles.
  const seatGeo = mergeGeometries([
    dropFaces(new THREE.BoxGeometry(1.6, 0.3, 1.3), [3, 4]).translate(0, 1.2, 0),
    dropFaces(new THREE.BoxGeometry(1.6, 1.3, 0.25), [3]).translate(0, 1.9, 0.6),
  ]);
  const seatMesh = new THREE.InstancedMesh(seatGeo, addHaze(makeToonMaterial(0xffffff)), seats.length);
  seats.forEach((s, i) => { place(seatMesh, i, [s.p[0], s.h, -s.p[1]], s.yaw); seatMesh.setColorAt(i, c.set(s.color)); });
  seatMesh.castShadow = seatMesh.receiveShadow = true;

  // Fans. Body base at local y 0; seated fans start at the seat (1.3), standing ones at the tread, 2.3x taller.
  const n = people.length;
  const bodyGeo = new THREE.BoxGeometry(1.25, 1.7, 0.9).translate(0, 0.85, 0.15);
  const headGeo = new THREE.SphereGeometry(0.42, 5, 3);
  const capGeo = new THREE.OctahedronGeometry(0.5, 0).scale(1, 0.62, 1).translate(0, 0.14, 0);
  const armGeo = new THREE.BoxGeometry(0.3, 1.5, 0.3).translate(0, 0.75, 0);
  const arms = people.filter(([, f]) => f.armsUp).length * 2;
  const bob = (len) => new Float32Array(len * 2);
  const bBody = bob(n), bHead = bob(n), bCap = bob(n), bArm = bob(arms);
  const body = addInstances(bodyGeo, bobMaterial(uTime, 0.5), n, bBody);
  const head = addInstances(headGeo, bobMaterial(uTime, 1), n, bHead);
  const hat = addInstances(capGeo, bobMaterial(uTime, 1), n, bCap);
  const arm = addInstances(armGeo, bobMaterial(uTime, 1.6), arms, bArm);
  let ai = 0;
  people.forEach(([s, f], i) => {
    const sit = f.standing ? 0.15 : 1.3, k = f.standing ? 2.3 : 0.9 + 0.2 * R();
    const yaw = s.yaw + (R() - 0.5) * (f.standing ? 0.2 : 0.4);
    const amp = (f.standing ? 0.45 : 0.1) * (0.6 + 0.8 * R()), ph = R();
    place(body, i, [s.p[0], s.h + sit, -s.p[1]], yaw, [1, k, 1]);
    const top = sit + 1.7 * k + 0.42;
    place(head, i, [s.p[0], s.h + top, -s.p[1]], yaw);
    place(hat, i, [s.p[0], s.h + top + 0.03, -s.p[1]], yaw);
    body.setColorAt(i, c.set(f.shirt));
    head.setColorAt(i, c.set(f.skin));
    hat.setColorAt(i, c.set(f.hair));
    for (const b of [bBody, bHead, bCap]) { b[2 * i] = amp; b[2 * i + 1] = ph; }
    if (f.armsUp) {
      for (const side of [-1, 1]) {
        // Shoulder 0.65 ft to the side of the torso's top; the arm leans outward 12-30 degrees.
        Qt.setFromAxisAngle(YAXIS, yaw);
        Vp.set(side * 0.7, 0, 0).applyQuaternion(Qt).add(Vs.set(s.p[0], s.h + sit + 1.7 * k - 0.35, -s.p[1]));
        Qz.setFromAxisAngle(ZAXIS, -side * (0.2 + 0.3 * R()));
        arm.setMatrixAt(ai, Mt.compose(Vp, Qt.clone().multiply(Qz), Vs.set(1, 1, 1)));
        arm.setColorAt(ai, c.set(f.skin));
        bArm[2 * ai] = amp; bArm[2 * ai + 1] = ph;
        ai++;
      }
    }
  });
  for (const m of [body, head, hat, arm]) m.receiveShadow = true;
  body.userData.tick = (t) => { uTime.value = t; };

  // Aisle stairs: a light strip on each aisle tread.
  const strip = new THREE.InstancedMesh(new THREE.BoxGeometry(3.6, 0.14, 2.8).translate(0, 0.07, 0), makeToonMaterial(0xf7efd9), stairs.length);
  stairs.forEach((s, i) => {
    place(strip, i, [s.p[0], s.h, -s.p[1]], s.yaw);
  });
  strip.receiveShadow = true;
  g.add(seatMesh, body, head, hat, arm, strip);
  g.userData.fanCount = n;
  return g;
}
