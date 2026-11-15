import * as THREE from "three";
import { JOINTS, J } from "./skeleton.js";
import { makePitcher, poseFromJoints, twoBone } from "./pitcher.js";
import { toScene } from "./coords.js";
import { PALETTE, makeToonMaterial } from "./look.js";

/*
 * The batter, catcher and home-plate umpire: the pitcher's stylised figure (pitcher.js) in hand-made
 * static poses, so the TV view shows the three people behind the plate that define the broadcast
 * shot. Muted uniform colours from the palette, no logos: the batter in road grey with a navy
 * helmet, the catcher in navy gear over home whites, the umpire in black shirt and cap, grey slacks.
 *
 * Each pose is written in the figure's own frame, the rest skeleton's (x = his left, y up,
 * z = the way he faces; feet at rest scale), then placed on the field. Knees and elbows come from
 * two-bone IK (pitcher.js twoBone), so the rig's fixed bone lengths are kept.
 */

const V = (x, y, z) => new THREE.Vector3(x, y, z);
const DEG = Math.PI / 180;
const REST = JOINTS.map((j) => V(...j.rest));
const seg = (a, b) => REST[J[a]].distanceTo(REST[J[b]]);
const UPPER = ["spine1", "spine2", "spine3", "neck", "left_collar", "right_collar", "left_shoulder", "right_shoulder"];

export const HEIGHT_IN = { batter: 74, catcher: 72, umpire: 73 };
export const BAT_FT = 2.8; // a 33-34 in bat

/*
 * Poses (rest-scale feet, figure frame). All are our own estimates of the textbook stances:
 * - batter: feet ~2.5 ft apart along the pitcher-catcher line, knees soft, trunk tipped ~20 deg over
 *   the plate, hands together just above and in front of the back shoulder, bat up and back.
 * - catcher: a full squat (pelvis ~1.5 ft up, feet wide, knees out), mitt held out as a low target
 *   at chest height in front of him, the bare hand tucked by his throwing-side knee.
 * - umpire: the "slot" stance, feet wide, knees bent, trunk ~35 deg forward, hands on his knees,
 *   head up, eyes ~4.75 ft over the field (cameras.js UMPIRE_EYE_HEIGHT_FT).
 */
const POSES = {
  batter: {
    pelvis: [0, 2.95, 0], flexDeg: 20, headKeep: 0.2,
    ankles: { left: [1.25, 0.28, 0.1], right: [-1.25, 0.28, -0.05] }, knees: { left: [0.3, 0, 1], right: [-0.3, 0, 1] },
    hands: { left: [-0.4, 4.85, 0.85], right: [-0.5, 4.95, 0.8] }, elbows: { left: [0.3, -1, 0.3], right: [-1, -0.6, 0] },
  },
  catcher: {
    pelvis: [0, 1.45, -0.2], flexDeg: 30, headKeep: 0.2,
    ankles: { left: [0.8, 0.28, 0.25], right: [-0.8, 0.28, 0.25] }, knees: { left: [0.7, 0.3, 1], right: [-0.7, 0.3, 1] },
    hands: { left: [0.25, 2.4, 1.3], right: [-0.6, 1.9, 0.7] }, elbows: { left: [1, -1, 0], right: [-1, -1, 0] },
  },
  umpire: {
    pelvis: [0, 2.7, -0.3], flexDeg: 35, headKeep: 0.3,
    ankles: { left: [0.85, 0.28, 0], right: [-0.85, 0.28, -0.25] }, knees: { left: [0.5, 0, 1], right: [-0.5, 0, 1] },
    hands: "knees", elbows: { left: [1, 0, 0], right: [-1, 0, 0] },
  },
};

/** The 24 joints (figure frame, rest-scale ft) of a pose. */
function localJoints(P) {
  const p = REST.map(() => null);
  const pelvis = V(...P.pelvis);
  const rel = (n, q) => REST[J[n]].clone().sub(REST[J.pelvis]).applyQuaternion(q).add(pelvis);
  const flex = new THREE.Quaternion().setFromAxisAngle(V(1, 0, 0), P.flexDeg * DEG); // lean forward (+z)
  p[J.pelvis] = pelvis;
  // Hips too: the rig's pelvis takes its "up" from spine1, so it tips with the lean (setBones).
  for (const n of [...UPPER, "left_hip", "right_hip"]) p[J[n]] = rel(n, flex);
  // Head kept up (eyes on the pitcher): it takes only part of the trunk's lean.
  const qHead = new THREE.Quaternion().setFromAxisAngle(V(1, 0, 0), P.flexDeg * P.headKeep * DEG);
  p[J.head] = REST[J.head].clone().sub(REST[J.neck]).applyQuaternion(qHead).add(p[J.neck]);
  for (const sd of ["left", "right"]) {
    const leg = twoBone(p[J[`${sd}_hip`]], V(...P.ankles[sd]), seg(`${sd}_hip`, `${sd}_knee`), seg(`${sd}_knee`, `${sd}_ankle`), V(...P.knees[sd]));
    p[J[`${sd}_knee`]] = leg.mid;
    p[J[`${sd}_ankle`]] = leg.end;
    p[J[`${sd}_foot`]] = leg.end.clone().add(REST[J[`${sd}_foot`]].clone().sub(REST[J[`${sd}_ankle`]])); // flat, forward
  }
  for (const sd of ["left", "right"]) {
    const target = P.hands === "knees" ? p[J[`${sd}_knee`]].clone().add(V(0, 0.25, 0.15)) : V(...P.hands[sd]);
    const fore = seg(`${sd}_elbow`, `${sd}_wrist`);
    const arm = twoBone(p[J[`${sd}_shoulder`]], target, seg(`${sd}_shoulder`, `${sd}_elbow`), fore + seg(`${sd}_wrist`, `${sd}_hand`), V(...P.elbows[sd]));
    p[J[`${sd}_elbow`]] = arm.mid;
    p[J[`${sd}_wrist`]] = arm.mid.clone().add(arm.end.clone().sub(arm.mid).setLength(fore));
    p[J[`${sd}_hand`]] = arm.end;
  }
  return p;
}

/*
 * Where each stands (Statcast ft; `side` = -1 for a right-handed batter, who stands on the catcher's
 * left, x < 0; +1 for a lefty):
 * - batter: pelvis 3.0 ft off the plate's centre line (box inner line 1.2 ft, OBR 2.01), level with
 *   the middle of the plate (y 0.6), facing the plate; his hands stay ~2 ft out, clear of any pitch
 *   over the plate.
 * - catcher: centred, pelvis 3 ft behind the plate's back point, facing the pitcher.
 * - umpire: in the slot between catcher and batter, 1 ft to the batter's side (the broadcast frames
 *   show him there), his head over cameras.js UMPIRE_Y_FT = -5 ft.
 */
function placement(kind, side) {
  if (kind === "batter") return { origin: [side * 3.0, 0.6], forward: [-side, 0] };
  if (kind === "catcher") return { origin: [0, -3.0], forward: [0, 1] };
  return { origin: [side * 1.0, -6.4], forward: [0, 1] };
}

/** Figure frame -> Statcast: left = up x forward (e.g. facing the pitcher, +y, his left is -x). */
function toStatcast(v, { origin, forward }, k) {
  const [fx, fy] = forward;
  const left = [-fy, fx];
  return [origin[0] + k * (v.x * left[0] + v.z * fx), origin[1] + k * (v.x * left[1] + v.z * fy), k * v.y];
}

/**
 * Joint positions (Statcast ft, JOINTS order) of `kind` ("batter" | "catcher" | "umpire") for batter
 * side `stand` ("R" | "L"); for the batter also the bat, {knob, tip}.
 */
export function castJoints(kind, stand) {
  const side = stand === "L" ? 1 : -1;
  const k = HEIGHT_IN[kind] / 75; // the rest skeleton is 75 in tall
  const place = placement(kind, side);
  const local = localJoints(POSES[kind]);
  const joints = local.map((v) => toStatcast(v, place, k));
  if (kind !== "batter") return { joints };
  // Bat: from the hands, up and back over the back shoulder (~45 deg), in the figure's frame.
  const knob = local[J.left_hand].clone().add(local[J.right_hand]).multiplyScalar(0.5);
  const tip = knob.clone().addScaledVector(V(-0.35, 0.85, -0.4).normalize(), BAT_FT / k);
  return { joints, bat: { knob: toStatcast(knob, place, k), tip: toStatcast(tip, place, k) } };
}

/**
 * Which figures each view shows. The umpire's view is the umpire's own eye and the batter's eye the
 * batter's, so each hides its own figure; the umpire's view also hides the catcher, whose crouched
 * head (~3.5 ft up, 2.5 ft in front of the eye) would cover the bottom of the zone and every low
 * pitch (the real umpire leans over him; this camera is on the centre line, 4.75 ft up).
 */
export function castVisibility(view) {
  return { batter: view !== "batter", catcher: view !== "umpire", umpire: view !== "umpire" };
}

const COLORS = {
  batter: { jersey: PALETTE.concreteShade, sleeve: PALETTE.navy, pants: PALETTE.concreteShade, cap: PALETTE.navy },
  catcher: { jersey: PALETTE.navy, sleeve: PALETTE.navy, pants: PALETTE.pants, cap: PALETTE.navy },
  umpire: { jersey: PALETTE.ink, sleeve: PALETTE.ink, pants: PALETTE.steel, cap: PALETTE.ink },
};

/** The three figures and the bat, posed for a right-handed batter; setCast re-poses and hides them. */
export function buildCast() {
  const g = new THREE.Group();
  g.name = "cast";
  const figs = {};
  for (const kind of ["batter", "catcher", "umpire"]) {
    // The catcher throws right-handed (glove on his left); batter and umpire have bare hands.
    figs[kind] = makePitcher("R", HEIGHT_IN[kind], { colors: COLORS[kind], glove: kind === "catcher" });
    g.add(figs[kind]);
  }
  const bat = new THREE.Mesh(new THREE.CylinderGeometry(0.11, 0.045, BAT_FT, 12), makeToonMaterial(PALETTE.trunk));
  bat.castShadow = true;
  g.add(bat);
  g.userData = { ...figs, bat, stand: null, view: null };
  return g;
}

/**
 * Pose the cast for batter side `stand` and show what `view` allows.
 * @returns {boolean} whether anything changed (the shadow map then needs a redraw).
 */
export function setCast(cast, stand, view) {
  const u = cast.userData;
  if (u.stand === stand && u.view === view) return false;
  if (u.stand !== stand) {
    for (const kind of ["batter", "catcher", "umpire"]) {
      poseFromJoints(u[kind], castJoints(kind, stand).joints.map(toScene));
    }
    const { knob, tip } = castJoints("batter", stand).bat;
    const a = toScene(knob), b = toScene(tip);
    u.bat.position.copy(a).add(b).multiplyScalar(0.5);
    u.bat.quaternion.setFromUnitVectors(V(0, 1, 0), b.clone().sub(a).normalize()); // barrel (wide end) at the tip
  }
  const vis = castVisibility(view);
  u.batter.visible = u.bat.visible = vis.batter;
  u.catcher.visible = vis.catcher;
  u.umpire.visible = vis.umpire;
  Object.assign(u, { stand, view });
  return true;
}
