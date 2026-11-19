import * as THREE from "three";
import { mergeGeometries } from "three/examples/jsm/utils/BufferGeometryUtils.js";
import { PALETTE, makeToonMaterial, fanHook } from "../look.js";
import { rng } from "./layout.js";
import { place } from "./build.js";
import { addHaze } from "./haze.js";
import { isOld } from "./old.js";
import { personGeometry, farArmsGeometry, FAN_GLSL, STYLE, MODE, STAND_HIP_FT } from "./person.js";

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
 *
 * Fans (default; ?old=fans restores the box fans above): each is a low-poly person (person.js): round
 * head with hair or cap, neck, shirt torso, two arms. One instanced mesh per LOD carries per-instance
 * skin, shirt, hair, style and arm behaviour; the vertex shader swings the arms (rest, clap in bursts,
 * cheer) and the outline pre-pass runs the same code (look.js fanHook), so ink follows the arms.
 * LOD: "near" fans (lower bowl and the first deck behind home, within NEAR_FT) are detailed
 * (head 8x5, eyes, arms); the rest are simple (6x3 head, plain torso, arms only for fans whose arms move).
 * Lighter-graphics mode draws the simple version everywhere.
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
export function planFans(seats, seed = 11, { v2 = false } = {}) {
  const R = rng(seed), R2 = rng(seed + 101); // R2 feeds only the person look, so v2:false is exactly the old crowd
  const wpick = (list) => { let x = R2() * list.reduce((a, [, w]) => a + w, 0); for (const [v, w] of list) if ((x -= w) < 0) return v; return list.at(-1)[0]; };
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
    const fan = {
      team, shirt: pick(shirts), skin: pick(SKINS), standing, armsUp: standing && R() < 0.65,
      hair: cap ? (team === "away" ? 0x5d3b8c : team === "home" ? 0xc9402f : 0x2a4a6b) : pick(HAIRS), cap,
    };
    return v2 ? personLook(fan, wpick, R2) : fan;
  });
}

// Realistic spreads (a ballpark crowd, not a palette swatch): common tones common, rare ones rare.
const SKIN_WEIGHTS = [[0xf6d5b8, 0.2], [0xf1c7a1, 0.22], [0xe0ac84, 0.18], [0xc68863, 0.14], [0xa8704a, 0.1], [0x8a5a3a, 0.08], [0x5e3b28, 0.05], [0x3d2a1e, 0.03]];
const HAIR_WEIGHTS = [[0x1a1a1a, 0.2], [0x2a1d16, 0.2], [0x4a3324, 0.2], [0x7d4a2a, 0.07], [0xd8b45a, 0.11], [0x8a6a3c, 0.08], [0xb0b0b0, 0.1], [0xe4e4e4, 0.04]];
// Team palettes with some off-team clothing (about 20%: grey, white, navy, denim) so no section looks uniformed.
const OFF_TEAM = [[0xdcdcdc, 3], [0xffffff, 3], [0x2a4a6b, 3], [0x4a6a8a, 2], [0x3c3c44, 2]];
const teamShirts = (list) => list.map((c, i) => [c, i < 2 ? 3 : 1.5]).concat(OFF_TEAM.map(([c, w]) => [c, w * 0.3]));

/** The person-specific look of one planned fan: skin, hair, shirt, hair style, arm behaviour. */
function personLook(fan, wpick, R2) {
  const shirts = fan.team === "neutral" ? NEUTRAL_SHIRTS.map((c) => [c, 1]) : teamShirts(fan.team === "away" ? AWAY_SHIRTS : HOME_SHIRTS);
  const skin = wpick(SKIN_WEIGHTS), shirt = wpick(shirts);
  const hairRoll = R2(), styleRoll = R2(), modeRoll = R2(), kid = R2() < 0.08;
  const hair = wpick(HAIR_WEIGHTS);
  const style = fan.cap ? STYLE.CAP : styleRoll < 0.07 ? STYLE.BALD : styleRoll < 0.37 ? STYLE.LONG : STYLE.SHORT;
  const mode = fan.standing ? (fan.armsUp ? MODE.CHEER : MODE.CLAP) : modeRoll < 0.03 ? MODE.CHEER : modeRoll < 0.28 ? MODE.CLAP : MODE.REST;
  void hairRoll;
  return { ...fan, skin, shirt, style, mode, kid,
    hair: style === STYLE.CAP ? fan.hair : style === STYLE.BALD ? new THREE.Color(skin).multiplyScalar(0.92).getHex() : hair };
}

/** Within this many ft of home plate (lower bowl, first deck) a fan gets the detailed mesh. */
export const NEAR_FT = 120;
/** "near" (detailed person) or "far" (simple): by deck and distance from home, fixed at build time. */
export const fanLod = (s) => (s.deck !== "bleacher" && Math.hypot(s.p[0], s.p[1]) < NEAR_FT ? "near" : "far");

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

/** ?old=fans: the box fans (body, hexagonal head, cap, raised arms for standing fans). */
function boxFans(people, R, uTime, c) {
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
  return [body, head, hat, arm];
}

const white = (hex) => new THREE.Color(hex);

/** Toon material for the person meshes: part colours from per-instance attributes (see person.js), animated by FAN_GLSL. */
function fanMaterial(uTime) {
  const m = makeToonMaterial(0xffffff, { surface: "none" });
  const park = m.onBeforeCompile;
  m.onBeforeCompile = (s, r) => {
    park(s, r);
    s.uniforms.uTime = uTime;
    s.vertexShader = s.vertexShader
      .replace("#include <common>", `#include <common>\nattribute vec3 aShirt;\nattribute vec3 aHair;\n${FAN_GLSL.common}`)
      .replace("#include <beginnormal_vertex>", `#include <beginnormal_vertex>\n${FAN_GLSL.normal}`)
      .replace("#include <begin_vertex>", `#include <begin_vertex>\n${FAN_GLSL.vertex}`)
      // instanceColor (skin) arrives in vColor; the other parts pick their own colour.
      .replace("#include <color_vertex>", `#include <color_vertex>
        { float pt = aPart.x; vec3 c = vColor;
          if (pt > 0.5 && pt < 1.5) c = aShirt;
          else if (pt > 1.5 && pt < 3.5) c = aHair;
          else if (pt > 3.5 && pt < 4.5) c = aHair * 0.8;
          else if (pt > 4.5 && pt < 5.5) c = vec3(0.012, 0.01, 0.01);
          else if (pt > 5.5) c = mix(vec3(0.02, 0.05, 0.14), vec3(0.2, 0.17, 0.12), step(0.5, fract(aAnim.y * 7.0)));
          vColor = c; }`);
  };
  addHaze(m);
  m.customProgramCacheKey = () => "crowdfan";
  return m;
}

/**
 * An instanced mesh of people. recs: per-fan records (fanRecords); simple: place them with the
 * simple-LOD transform (a standing fan has no legs there: its torso is stretched instead).
 */
function personMesh(geo, mat, recs, simple) {
  const n = recs.length, g = geo.clone();
  const mesh = new THREE.InstancedMesh(g, mat, n);
  const shirt = new Float32Array(3 * n), hair = new Float32Array(3 * n), anim = new Float32Array(4 * n);
  recs.forEach((f, i) => {
    const k = f.k;
    place(mesh, i, [f.x, f.hip(simple), f.z], f.yaw, [k, simple && f.standing ? 1.9 * k : k, k]);
    mesh.setColorAt(i, white(f.skin));
    white(f.shirt).toArray(shirt, 3 * i);
    white(f.hair).toArray(hair, 3 * i);
    anim.set([f.amp, f.ph, f.mode, f.style + 2], 4 * i);
  });
  g.setAttribute("aShirt", new THREE.InstancedBufferAttribute(shirt, 3));
  g.setAttribute("aHair", new THREE.InstancedBufferAttribute(hair, 3));
  g.setAttribute("aAnim", new THREE.InstancedBufferAttribute(anim, 4));
  mesh.receiveShadow = true;
  return mesh;
}

/**
 * Per-fan records for the person meshes: where, how big (kids a little smaller), and the look and
 * animation attributes. people: [[seat, plannedFan]]; R: rng. Pure data (tested).
 */
export function fanRecords(people, R) {
  return people.map(([s, f]) => {
    const k = (f.kid ? 0.78 : 0.92) + 0.16 * R(), yaw = s.yaw + (R() - 0.5) * (f.standing ? 0.2 : 0.4);
    const amp = (f.standing ? 0.45 : 0.1) * (0.6 + 0.8 * R()), ph = R();
    return {
      x: s.p[0], z: -s.p[1], yaw, k, amp, ph, standing: f.standing, mode: f.mode, style: f.style,
      skin: f.skin, shirt: f.shirt, hair: f.hair, near: fanLod(s) === "near",
      // Hip height: seated on the pan (1.3 ft); standing: legs under the hips (detailed) or a stretched torso (simple).
      hip: (simple) => s.h + (f.standing ? (simple ? 0.7 : STAND_HIP_FT * k) : 1.3 * (0.5 + 0.5 * k)),
    };
  });
}

/** The person meshes (detailed near, simple far, and the lighter-graphics stand-ins); tick on `group` swaps them. */
function personFans(people, R, uTime, group) {
  const recs = fanRecords(people, R);
  const mat = fanMaterial(uTime);
  const near = recs.filter((r) => r.near), far = recs.filter((r) => !r.near);
  const moving = (a) => a.filter((r) => r.mode > 0);
  const sets = [
    [personGeometry("near"), near.filter((r) => !r.standing), false, "detail"],
    [personGeometry("near", { legs: true }), near.filter((r) => r.standing), false, "detail"],
    [personGeometry("far"), far, true, "far"],
    [farArmsGeometry(), moving(far), true, "far"],
    [personGeometry("far"), near, true, "lite"], // swapped in for the detailed meshes in lighter-graphics mode
    [farArmsGeometry(), moving(near), true, "lite"],
  ].filter(([, list]) => list.length);
  const meshes = sets.map(([geo, list, simple, lod]) => {
    const m = personMesh(geo, mat, list, simple);
    m.userData.lod = lod;
    return m;
  });
  let lite = null;
  const show = () => {
    if (lite === fanHook.lite) return;
    lite = fanHook.lite;
    for (const m of meshes) m.visible = m.userData.lod === "far" || (m.userData.lod === "lite") === lite;
  };
  show();
  group.userData.tick = (t) => { uTime.value = t; show(); };
  fanHook.glsl = FAN_GLSL; // the outline pre-pass animates the fans too (look.js)
  fanHook.uTime = uTime;
  return meshes;
}

/**
 * Seats, fans and stairs as instanced meshes.
 * @param seats [{p, h, yaw, color, deck, row, rows, sec}] (stands.js)
 * @param stairs [{p, h, yaw}] one per aisle tread: a lighter step strip
 * @returns Group
 */
export function buildCrowd(seats, stairs, seed = 11) {
  const g = new THREE.Group();
  const oldFans = isOld("fans");
  const R = rng(seed + 1);
  const people = planFans(seats, seed, { v2: !oldFans }).map((f, i) => [seats[i], f]).filter(([, f]) => f);
  const c = new THREE.Color();
  const uTime = { value: 0 };

  // Seats: pan and back; the back is on the side away from the field (+z local). The person fans' seat backs
  // are taller: they show beside the head and shoulders, and between people where a seat is empty.
  // Hidden faces dropped (BoxGeometry face order: +x -x +y -y +z -z): pan's bottom and back, the back's bottom. 24 -> 18 triangles.
  const seatGeo = mergeGeometries([
    dropFaces(new THREE.BoxGeometry(1.6, 0.3, 1.3), [3, 4]).translate(0, 1.2, 0),
    oldFans ? dropFaces(new THREE.BoxGeometry(1.6, 1.3, 0.25), [3]).translate(0, 1.9, 0.6)
      : dropFaces(new THREE.BoxGeometry(1.6, 1.9, 0.25), [3]).translate(0, 2.15, 0.6),
  ]);
  const seatMesh = new THREE.InstancedMesh(seatGeo, addHaze(makeToonMaterial(0xffffff)), seats.length);
  seats.forEach((s, i) => { place(seatMesh, i, [s.p[0], s.h, -s.p[1]], s.yaw); seatMesh.setColorAt(i, c.set(s.color)); });
  seatMesh.castShadow = seatMesh.receiveShadow = true;

  // Aisle stairs: a light strip on each aisle tread.
  const strip = new THREE.InstancedMesh(new THREE.BoxGeometry(3.6, 0.14, 2.8).translate(0, 0.07, 0), makeToonMaterial(0xf7efd9), stairs.length);
  stairs.forEach((s, i) => {
    place(strip, i, [s.p[0], s.h, -s.p[1]], s.yaw);
  });
  strip.receiveShadow = true;
  g.add(seatMesh, ...(oldFans ? boxFans(people, R, uTime, c) : personFans(people, R, uTime, g)), strip);
  g.userData.fanCount = people.length;
  return g;
}
