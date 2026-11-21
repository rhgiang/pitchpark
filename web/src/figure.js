import * as THREE from "three";
import { mergeGeometries } from "three/examples/jsm/utils/BufferGeometryUtils.js";
import { JOINTS, J } from "./skeleton.js";
import { PALETTE, makeToonMaterial } from "./look.js";
import { handModel } from "./handModels.js";

/**
 * The pitcher's meshes (M5 look): a stylised, toon-shaded ballplayer built from simple shapes on
 * the SMPL skeleton (skeleton.js), in its rest T-pose (scene ft, 75 in tall, x = the figure's
 * left, y up, z = facing). Every part is rigidly bound to one bone, except the torso, whose
 * vertices blend between neighbouring spine bones so the back bends without a crease.
 *
 * Why the joints never gap: each limb is a "tapered capsule" (the hull of two spheres) whose end
 * spheres sit exactly on the joints, and the next limb starts with a sphere of the same radius on
 * the same joint. A sphere looks the same from every angle, so the elbow, knee, shoulder and
 * hip stay closed whatever the bend, like a ball joint on an action figure.
 *
 * Parts are merged per material: one draw call per material (about 8), not one per part.
 */

const V = (x, y, z) => new THREE.Vector3(x, y, z);
const R = (name) => new THREE.Vector3(...JOINTS[J[name]].rest);
const UP = V(0, 1, 0);

// Leather tones for the mitt (darker lacing/binding, lighter oiled pocket), around PALETTE.glove.
const GLOVE_DARK = 0x5b3a1c;
const GLOVE_LIGHT = 0xb27a43;
const HAIR = 0x2b1d14; // dark brown, showing under the cap at the back and sides
const COLORS = {
  jersey: PALETTE.jersey, navy: PALETTE.navy, pants: PALETTE.pants, skin: PALETTE.skin, hair: HAIR,
  glove: PALETTE.glove, gloveDark: GLOVE_DARK, gloveLight: GLOVE_LIGHT, shoe: PALETTE.shoe,
  capDark: 0x1b3049, rope: PALETTE.coral, // the sculpted cap's seams/stitching and front rope (make_pitcher.py)
};

/**
 * Bind a geometry to bones. `boneOf(restPos)` gives a bone index (rigid) or [[bone, weight], ...]
 * (up to 4, weights summing to 1).
 */
function skin(geo, boneOf) {
  const pos = geo.attributes.position;
  const idx = new Uint16Array(pos.count * 4);
  const wt = new Float32Array(pos.count * 4);
  const v = new THREE.Vector3();
  for (let i = 0; i < pos.count; i++) {
    const b = boneOf(v.fromBufferAttribute(pos, i));
    (typeof b === "number" ? [[b, 1]] : b).forEach(([bi, w], n) => { idx[i * 4 + n] = bi; wt[i * 4 + n] = w; });
  }
  geo.setAttribute("skinIndex", new THREE.Uint16BufferAttribute(idx, 4));
  geo.setAttribute("skinWeight", new THREE.Float32BufferAttribute(wt, 4));
  return geo;
}
const on = (bone) => (g) => skin(g, () => J[bone]);

/** Reverse every triangle (inside out). With `normals`, also turn the normals round. */
function flip(g, normals = false) {
  const ix = g.index.array;
  for (let i = 0; i < ix.length; i += 3) [ix[i + 1], ix[i + 2]] = [ix[i + 2], ix[i + 1]];
  if (normals) { const n = g.attributes.normal.array; for (let i = 0; i < n.length; i++) n[i] = -n[i]; }
  return g;
}
/** Mirror in x (left <-> right). A mirror turns triangles inside out, so they are flipped back. */
const mirrorX = (g) => flip(g.scale(-1, 1, 1));

/**
 * Tapered capsule from a (radius ra) to b (radius rb): two spheres joined by the cone tangent to
 * both. The tangent touches each sphere at elevation asin((ra - rb) / L) from its equator.
 */
function taper(a, b, ra, rb = ra, radial = 12) {
  const L = Math.max(a.distanceTo(b), 1e-3);
  const al = Math.asin(Math.max(-1, Math.min(1, (ra - rb) / L)));
  const pts = [];
  const N = 4;
  for (let i = 0; i <= N; i++) { const p = -Math.PI / 2 + ((al + Math.PI / 2) * i) / N; pts.push(new THREE.Vector2(Math.max(ra * Math.cos(p), 0), ra * Math.sin(p))); }
  for (let i = 0; i <= N; i++) { const p = al + ((Math.PI / 2 - al) * i) / N; pts.push(new THREE.Vector2(Math.max(rb * Math.cos(p), 0), L + rb * Math.sin(p))); }
  const g = new THREE.LatheGeometry(pts, radial);
  g.applyQuaternion(new THREE.Quaternion().setFromUnitVectors(UP, b.clone().sub(a).normalize()));
  return g.translate(a.x, a.y, a.z);
}

/** Ellipsoid with semi-axes (sx, sy, sz) at c. */
const blob = (c, sx, sy, sz, w = 16, h = 10) =>
  new THREE.SphereGeometry(1, w, h).scale(sx, sy, sz).translate(c.x, c.y, c.z);

/** A lathe (radius, height) profile, squashed to an oval body cross-section (sx wide, sz deep). */
const lathe = (profile, sx, sz, seg = 28) =>
  new THREE.LatheGeometry(profile.map(([r, y]) => new THREE.Vector2(r, y)), seg).scale(sx, 1, sz);

const smooth = (x) => (x <= 0 ? 0 : x >= 1 ? 1 : x * x * (3 - 2 * x));
/**
 * Torso skinning: a vertex belongs to the spine bone of its height band, blended into the next
 * bone over +-0.1 ft around each spine joint, so a bent back is a curve, not a hinge.
 */
function spineWeights(v) {
  const chain = [J.pelvis, J.spine1, J.spine2, J.spine3];
  const cuts = [R("spine1").y, R("spine2").y, R("spine3").y];
  const H = 0.1;
  for (let i = 0; i < cuts.length; i++) {
    if (v.y < cuts[i] - H) return chain[i];
    if (v.y < cuts[i] + H) { const w = smooth((v.y - cuts[i] + H) / (2 * H)); return [[chain[i], 1 - w], [chain[i + 1], w]]; }
  }
  return J.spine3;
}

// ---------- hands (built for the LEFT side, +x outward from the wrist; mirrored for the right) ----------
// Local frame at the wrist in the T-pose: +x along the hand, +y the back of the hand (palm down),
// +z the thumb side (forward).

/**
 * The throwing hand holding the ball. The ball's centre is the knuckle joint (SMPL "hand", what
 * handPosition returns), 0.28 ft from the wrist. So the palm is shortened to stop at the ball's
 * surface, and three fingers (a cartoon hand: thumb + 3) curl over its front at the ball's radius
 * plus a finger's, with the thumb underneath: the ball sits in the grip, not in the palm.
 */
function gripHand() {
  const K = V(0.28, 0, 0); // ball centre = knuckles
  const BALL = 1.45 / 12; // ball.js BALL_RADIUS_FT (not imported: keeps this module three-only)
  const parts = [blob(V(0.075, 0, 0.005), 0.095, 0.055, 0.1, 12, 8)]; // palm, ends at the ball
  // A finger at depth z wraps the ball's cross-section there: circle radius sqrt(B^2 - z^2).
  const finger = (z, r, degs) => {
    const rad = Math.sqrt(BALL * BALL - z * z) + r;
    const p = degs.map((d) => V(K.x + rad * Math.cos(d * Math.PI / 180), rad * Math.sin(d * Math.PI / 180), z));
    for (let i = 0; i + 1 < p.length; i++) parts.push(taper(p[i], p[i + 1], r, r * 0.92, 8));
  };
  finger(0.04, 0.027, [158, 105, 55, 10]); // index
  finger(-0.025, 0.028, [160, 105, 52, 5]); // middle
  finger(-0.075, 0.025, [165, 115, 70]); // ring (pinky tucked: cartoon 4-digit hand)
  // Thumb: from the palm's thumb side, under the ball, tip coming up its far side.
  const tz = 0.06, tr = Math.sqrt(BALL * BALL - tz * tz) + 0.03;
  const at = (d) => V(K.x + tr * Math.cos(d * Math.PI / 180), tr * Math.sin(d * Math.PI / 180), tz);
  parts.push(taper(V(0.05, -0.02, 0.07), at(205), 0.034, 0.031, 8), taper(at(205), at(255), 0.031, 0.029, 8),
    taper(at(255), at(300), 0.029, 0.026, 8));
  return parts;
}

/**
 * Fielder's mitt on the glove hand. One lathe surface, swept round and squashed to an oval: the
 * domed back, the rolled edge (binding, darker: the lacing), and the cupped pocket opening to
 * the palm side (-y), lighter like oiled leather. Then four finger stalls over the back, a thumb
 * stall on the +z edge, a webbing panel between thumb and index, and a wrist cuff.
 * Real pitcher's gloves are ~11.5-12 in long; this one is 0.95 ft from heel to finger tips.
 * @returns {[string, THREE.BufferGeometry][]} material key, geometry
 */
function mitt() {
  // Profile from the pocket's centre outwards, round the rim, then over the back to its top, so
  // the surface faces out of the leather everywhere (r normalised to 1 at the rim).
  const pocket = [[0, 0.03], [0.35, 0.025], [0.65, 0.005], [0.85, -0.02], [0.93, -0.035]];
  const rim = [[0.93, -0.035], [0.99, -0.045], [1.04, -0.025], [1.03, 0.005], [1.0, 0.02]];
  const back = [[1.0, 0.02], [0.95, 0.06], [0.8, 0.1], [0.55, 0.125], [0.25, 0.135], [0, 0.138]];
  const shell = (p) => lathe(p, 0.46, 0.3, 24).translate(0.47, 0, 0);
  const out = [["gloveLight", shell(pocket)], ["gloveDark", shell(rim)], ["glove", shell(back)]];
  // Finger stalls: rounded tubes over the back, fanned slightly, tips past the rim.
  for (const [z, zt] of [[0.13, 0.17], [0.045, 0.06], [-0.04, -0.05], [-0.12, -0.15]]) {
    out.push(["glove", taper(V(0.45, 0.1, z), V(0.96, 0.01, zt), 0.062, 0.055, 10)]);
  }
  out.push(["glove", taper(V(0.1, 0.03, 0.24), V(0.62, 0.0, 0.36), 0.075, 0.062, 10)]); // thumb stall
  // Webbing between thumb tip and index stall: a flat, darker panel.
  out.push(["gloveDark", blob(V(0, 0, 0), 0.16, 0.03, 0.085, 12, 6).rotateY(-0.35).translate(0.8, 0, 0.29)]);
  // Wrist cuff: a short dark band where the hand enters, over the forearm's end.
  out.push(["gloveDark", new THREE.CylinderGeometry(0.15, 0.16, 0.13, 14).rotateZ(Math.PI / 2).translate(0.02, 0, 0)]);
  return out;
}
/** Put a left-built hand part on `side`'s wrist (mirrored for the right). */
function atWrist(g, side) {
  if (side === "right") mirrorX(g);
  const w = R(`${side}_wrist`);
  return g.translate(w.x, w.y, w.z);
}

// ---------- the body ----------

/**
 * All parts, as [material key, skinned geometry]. Proportions for a 6 ft 3 in athlete at rest
 * scale (the Group scales to each pitcher's height): broad shoulders (~1.65 ft across the
 * deltoids), a V-taper to a ~1.1 ft waist, solid thighs.
 */
function bodyParts(throws) {
  const G = throws === "L" ? "right" : "left";
  const parts = [];
  // A sculpted body (pitcher.glb, already skinned: handModels.js) replaces the shapes below when loaded.
  const model = handModel("pitcher");
  if (model) parts.push(...model.map(([m, g]) => [m in COLORS ? m : "jersey", g]));
  else proceduralBody(parts);
  for (const sd of ["left", "right"]) {
    // Hand: the mitt on the glove side, the ball grip on the throwing side.
    const wrist = (g) => skin(atWrist(g, sd), () => J[`${sd}_wrist`]);
    // A sculpted model (handModels.js) replaces the procedural hand when it has loaded.
    const hand = sd === G ? handModel("mitt") ?? mitt() : handModel("grip") ?? gripHand().map((g) => ["skin", g]);
    for (const [m, g] of hand) parts.push([m in COLORS ? m : "glove", wrist(g)]);
  }
  return parts;
}

/** The body built from simple shapes (no hands), pushed onto `parts` as [material key, skinned geometry]. */
function proceduralBody(parts) {
  const add = (mat, g, boneOf) => parts.push([mat, skin(g, boneOf)]);

  // Torso: jersey from the belt to the shoulders, widest across the chest/shoulders.
  add("jersey", lathe([[0, 3.38], [0.42, 3.4], [0.41, 3.7], [0.43, 4.05], [0.49, 4.45], [0.53, 4.8],
    [0.53, 4.98], [0.45, 5.16], [0.26, 5.3], [0, 5.33]], 1.32, 0.7), spineWeights);
  // Pants seat and hips (on the pelvis), with a navy belt band at the waist.
  add("pants", lathe([[0, 2.82], [0.34, 2.86], [0.46, 3.05], [0.47, 3.3], [0.45, 3.52], [0, 3.56]], 1.25, 0.76), () => J.pelvis);
  add("navy", new THREE.CylinderGeometry(0.474, 0.47, 0.09, 28, 1, true).scale(1.27, 1, 0.78).translate(0, 3.47, 0), () => J.pelvis);
  // Navy collar ring round the neck opening.
  add("navy", new THREE.TorusGeometry(0.17, 0.03, 6, 20).rotateX(Math.PI / 2).scale(1, 1, 0.85).translate(0, 5.27, -0.03), () => J.spine3);

  // Neck and head: a slightly egg-shaped head (taller than wide), ears to show which way he
  // faces, a small nose; no face otherwise (face-agnostic: any pitcher).
  add("skin", taper(R("neck"), R("head"), 0.15, 0.14), () => J.neck);
  const H = V(0, 5.86, 0.04); // head centre
  add("skin", blob(H, 0.32, 0.37, 0.35, 22, 14), () => J.head);
  for (const x of [-0.31, 0.31]) add("skin", blob(V(x, 5.84, 0.02), 0.04, 0.08, 0.06, 8, 6), () => J.head);
  add("skin", blob(V(0, 5.8, 0.39), 0.045, 0.06, 0.05, 8, 6), () => J.head);
  add("hair", blob(V(0, 5.88, -0.08), 0.335, 0.26, 0.31), () => J.head); // back and sides, under the cap
  // Cap: crown dome, a button on top, and a brim curving down a little at the front edge.
  add("navy", new THREE.SphereGeometry(0.37, 22, 8, 0, 2 * Math.PI, 0, Math.PI / 2)
    .scale(1, 0.85, 1.02).translate(0, 5.92, 0.03), () => J.head);
  add("navy", blob(V(0, 6.235, 0.03), 0.03, 0.015, 0.03, 8, 4), () => J.head);
  // The brim is a shallow slice of a big sphere (rim radius 0.55 sin 0.5 = 0.26 ft), with an
  // inside-out copy just under it so it is solid seen from below too.
  const brim = () => new THREE.SphereGeometry(0.55, 20, 3, 0, 2 * Math.PI, 0, 0.5).scale(1, 0.5, 0.95)
    .translate(0, -0.241, 0).rotateX(0.1).translate(0, 5.89, 0.33);
  add("navy", brim(), () => J.head);
  add("navy", flip(brim(), true).translate(0, -0.012, 0), () => J.head);

  for (const sd of ["left", "right"]) {
    const x = sd === "left" ? 1 : -1; // outward
    const P = (n) => R(`${sd}_${n}`);
    // Shoulder cap (deltoid) on the collar bone: always centred on the shoulder joint.
    add("jersey", blob(P("shoulder"), 0.215, 0.21, 0.2, 14, 10), () => J[`${sd}_collar`]);
    // Upper arm (navy undershirt sleeve) with the jersey's short sleeve over its top.
    add("navy", taper(P("shoulder"), P("elbow"), 0.17, 0.13), () => J[`${sd}_shoulder`]);
    add("jersey", taper(P("shoulder"), P("shoulder").lerp(P("elbow"), 0.42), 0.215, 0.185), () => J[`${sd}_shoulder`]);
    // Forearm: thicker below the elbow, slim at the wrist.
    add("skin", taper(P("elbow"), P("wrist"), 0.13, 0.095), () => J[`${sd}_elbow`]);

    // Thigh: thick at the hip, a sphere at the knee shared with the shin.
    add("pants", taper(P("hip"), P("knee"), 0.28, 0.19, 14), () => J[`${sd}_hip`]);
    // Navy piping down the outside seam of the pants.
    add("navy", taper(P("hip").add(V(x * 0.27, -0.05, 0)), P("knee").add(V(x * 0.185, 0.1, 0)), 0.022, 0.022, 6), () => J[`${sd}_hip`]);
    // Shin: knicker-length pants (bunched just below the knee), then a navy sock to the ankle.
    const calf = P("knee").lerp(P("ankle"), 0.45);
    add("pants", taper(P("knee"), calf, 0.2, 0.18), () => J[`${sd}_knee`]);
    add("navy", taper(P("knee"), P("ankle"), 0.17, 0.11), () => J[`${sd}_knee`]);
    // Shoe on the ankle bone: a rounded upper over a white sole, heel behind the ankle.
    const a = P("ankle");
    add("shoe", blob(V(a.x + x * 0.01, 0.15, 0.16), 0.15, 0.12, 0.38, 14, 8), () => J[`${sd}_ankle`]);
    add("shoe", taper(a, V(a.x, 0.12, -0.12), 0.115, 0.12, 10), () => J[`${sd}_ankle`]);
    add("jersey", blob(V(a.x + x * 0.01, 0.04, 0.15), 0.16, 0.04, 0.42, 14, 6), () => J[`${sd}_ankle`]);
  }
}

// The bone a part's mesh stretches toward when its bone is longer or shorter (first child, except
// where the first is a hip: pelvis -> spine1, spine3 -> neck).
const FIT_CHILD = JOINTS.map((_, i) => {
  const kids = JOINTS.flatMap((j, c) => (j.parent === i ? [c] : []));
  return i === J.pelvis ? J.spine1 : i === J.spine3 ? J.neck : (kids[0] ?? -1);
});

/**
 * Re-fit the rest-pose mesh to a pitcher's own bone lengths (rest from skeleton.js restFromLengths).
 * Each vertex moves by its bones' joint displacements, blended along the bone from its start joint
 * (0) to its end joint (1), so a longer shin is a longer shin and the sphere at the knee follows
 * the knee. Thickness is unchanged (a long-limbed pitcher is longer, not fatter). The wrist bone
 * moves rigidly with the hand joint: the ball sits on it, and the fingers must stay round it.
 */
function fitToRest(geometry, rest) {
  const delta = JOINTS.map((_, i) => rest[i].clone().sub(R(JOINTS[i].smpl)));
  const pos = geometry.attributes.position, si = geometry.attributes.skinIndex, sw = geometry.attributes.skinWeight;
  const v = new THREE.Vector3(), axis = new THREE.Vector3(), d = new THREE.Vector3();
  for (let i = 0; i < pos.count; i++) {
    v.fromBufferAttribute(pos, i);
    const move = new THREE.Vector3();
    for (let n = 0; n < 4; n++) {
      const w = sw.getComponent(i, n);
      if (w === 0) continue;
      const b = si.getComponent(i, n), c = FIT_CHILD[b];
      let t = 0;
      if (c >= 0) {
        const o = R(JOINTS[b].smpl);
        axis.copy(R(JOINTS[c].smpl)).sub(o);
        t = b === J.left_wrist || b === J.right_wrist ? 1 : Math.min(Math.max(v.clone().sub(o).dot(axis) / axis.lengthSq(), 0), 1);
      }
      move.addScaledVector(c >= 0 ? d.copy(delta[b]).lerp(delta[c], t) : delta[b], w);
    }
    pos.setXYZ(i, v.x + move.x, v.y + move.y, v.z + move.z);
  }
  return geometry;
}

/**
 * The figure's merged geometry (one group per material) and its materials, for `throws`
 * ("R" | "L": the mitt goes on the other hand). Materials are named by their COLORS key.
 * `rest` (optional): the pitcher's own rest joints; the default is the 75 in rig.
 */
export function figureGeometry(throws, rest = null) {
  const parts = bodyParts(throws);
  const keys = [...new Set(parts.map((p) => p[0]))];
  const byMat = keys.map((k) => mergeGeometries(parts.filter((p) => p[0] === k).map((p) => p[1])));
  const materials = keys.map((k) => Object.assign(makeToonMaterial(COLORS[k]), { name: k }));
  const geometry = mergeGeometries(byMat, true);
  return { geometry: rest ? fitToRest(geometry, rest) : geometry, materials };
}
