import * as THREE from "three";
import { mergeGeometries } from "three/examples/jsm/utils/BufferGeometryUtils.js";
import { JOINTS, J, buildSkeleton } from "./skeleton.js";
import { toScene } from "./coords.js";
import { groundHeightFt, RUBBER_FRONT_Y_FT } from "./field.js";
import { PALETTE, makeToonMaterial } from "./look.js";

const V = (x, y, z) => new THREE.Vector3(x, y, z);
const DEG = Math.PI / 180;
const REST = JOINTS.map((j) => V(...j.rest));
const CHILDREN = JOINTS.map((_, i) => JOINTS.flatMap((j, c) => (j.parent === i ? [c] : [])));
const segFt = (a, b) => REST[J[a]].distanceTo(REST[J[b]]); // at the 75 in rest size
const LEG_FT = segFt("left_hip", "left_knee") + segFt("left_knee", "left_ankle");
const FOOT_FT = segFt("left_ankle", "left_foot");
const ANKLE_UP_FT = REST[J.left_ankle].y; // ankle height above a flat-planted sole
/** Ground height under a scene point (scene z = -Statcast y). */
const groundY = (p) => groundHeightFt(p.x, -p.z);

// Release posture. Biomechanics values at ball release from Fleisig et al. (1999, J Biomech 32)
// and Escamilla et al. (1998, J Appl Biomech): elbow flexed ~20-25 deg, shoulder abducted
// ~90-100 deg, lead knee flexed ~35-50 deg, trunk flexed forward ~30-55 deg. Arm slot comes
// mostly from contralateral trunk tilt, not from more shoulder abduction.
const ELBOW_FLEX_DEG = 20;
const ABDUCTION_OVER_90_DEG = 10; // arm 10 deg above the shoulder line; trunk tilt does the rest
const MAX_TRUNK_FLEX_DEG = 65;
const TYPICAL_TRUNK_FLEX_DEG = 40;
const TYPICAL_STRIDE_FT = 5.6; // ~0.9 of height (Fleisig: stride 75-90% of height), rubber to lead ankle
const FRONT_LEG_REACH = 0.93; // front hip-to-ankle / leg length: lead knee flexed ~45 deg
const FRONT_FOOT_LEAD_FT = [0.6, 2.4]; // braced lead leg: front ankle this far ahead of its hip
const BACK_LEG_REACH = 0.97; // trailing leg nearly straight
const BACK_ANKLE_UP_FT = 0.45; // trailing foot drags on its toe, heel up
const PIVOT_FOOT_Y_FT = RUBBER_FRONT_Y_FT + 0.25; // the drag never goes behind the rubber

/**
 * Two-bone IK (law of cosines): the middle joint (elbow/knee) for a chain of lengths a, b from
 * `root` towards `target`, bending towards `pole`. If the target is out of reach the chain
 * straightens towards it instead of stretching. Returns {mid, end, reached}.
 */
function twoBone(root, target, a, b, pole) {
  const u = target.clone().sub(root);
  const want = u.length();
  const dist = Math.min(Math.max(want, Math.abs(a - b) + 1e-4), (a + b) * 0.9999);
  u.normalize();
  const cosA = (a * a + dist * dist - b * b) / (2 * a * dist);
  const w = pole.clone().addScaledVector(u, -pole.dot(u)).normalize();
  return {
    mid: root.clone().addScaledVector(u, a * cosA).addScaledVector(w, a * Math.sqrt(1 - cosA * cosA)),
    end: root.clone().addScaledVector(u, dist),
    reached: Math.abs(dist - want) < 1e-3,
  };
}

/** Ball of the foot: flat towards the plate if the ankle is low enough, else pointing down. */
function toeTarget(ankle, k = 1) {
  const foot = FOOT_FT * k;
  const dh = ankle.y - (groundY(ankle) + 0.06 * k);
  if (dh < foot * 0.95) return ankle.clone().add(V(0, -dh, Math.sqrt(foot ** 2 - dh * dh)));
  return ankle.clone().addScaledVector(V(0, -1, 0.3).normalize(), foot);
}

/**
 * World positions (scene ft) of all 24 joints at ball release, plus what had to give.
 *
 * Construction: the hand is pinned to the release point. The shoulder sits one arm reach away
 * (elbow flexed ELBOW_FLEX_DEG) along the arm angle in the catcher's-view plane, which is how
 * Statcast defines `arm_angle` (shoulder to ball, from horizontal, seen from behind the plate);
 * putting the shoulder at the hand's depth makes the 3D elevation equal it too. The upper body
 * is rigid: tilted towards the glove side so the shoulder line rises by (arm angle - 10 deg),
 * then flexed forward. Flexion and lead-foot placement are searched so the lead leg lands with
 * its knee bent about 45 deg, near a typical flexion and stride; the trailing foot drags on its
 * toe as far back towards the rubber as the leg reaches. When the release cannot be met with
 * that posture, the trunk and stride give way (more flexion, a longer stride; if needed the
 * lead foot floats or the knee bends more or less); bones never stretch, and the hand and arm
 * angle are never compromised.
 *
 * @param throws "R" | "L"
 * @param release Statcast [x, y, z] ft of the release point
 * @param armAngleDeg Statcast arm_angle: 0 sidearm, 90 straight over the top
 * @param k body scale, height_in / 75 (every length of the 75 in rest skeleton times k)
 * @returns {{targets: THREE.Vector3[], flexDeg: number, frontPlanted: boolean,
 *   kneeAsWanted: boolean, backGrounded: boolean, strideFt: number}}
 */
export function solvePose(throws, release, armAngleDeg, k = 1) {
  // A taller pitcher is the same body scaled up: bones, typical stride and foot heights all
  // scale with k; the field (rubber, mound) does not.
  const seg = (a, b) => k * segFt(a, b);
  const rest = (name) => REST[J[name]].clone().multiplyScalar(k);
  const s = throws === "L" ? 1 : -1; // x side of the throwing arm: a RHP releases at x < 0
  const T = throws === "L" ? "left" : "right";
  const G = throws === "L" ? "right" : "left"; // glove side = lead leg side
  const th = armAngleDeg * DEG;
  const hand = toScene(release);
  const a = seg(`${T}_shoulder`, `${T}_elbow`);
  const forearm = seg(`${T}_elbow`, `${T}_wrist`);
  const b = forearm + seg(`${T}_wrist`, `${T}_hand`); // wrist kept straight
  const reach = Math.sqrt(a * a + b * b + 2 * a * b * Math.cos(ELBOW_FLEX_DEG * DEG));
  const shoulder = hand.clone().sub(V(s * Math.cos(th), Math.sin(th), 0).multiplyScalar(reach));
  const tilt = Math.max(0, armAngleDeg - ABDUCTION_OVER_90_DEG) * DEG;

  // Rigid upper body for a forward flexion: rotate about +x (flex towards the plate), then about
  // +z (forward) by s * tilt, which raises the throwing shoulder. Pelvis placed so the throwing
  // shoulder lands on `shoulder`.
  const trunk = (flex) => {
    const q = new THREE.Quaternion().setFromAxisAngle(V(0, 0, 1), s * tilt)
      .multiply(new THREE.Quaternion().setFromAxisAngle(V(1, 0, 0), flex));
    const pelvis = shoulder.clone().sub(rest(`${T}_shoulder`).sub(rest("pelvis")).applyQuaternion(q));
    const at = (name) => rest(name).sub(rest("pelvis")).applyQuaternion(q).add(pelvis);
    return { q, at, flex };
  };
  const frontAnkle = (hip, lead) => {
    const p = V(hip.x, 0, hip.z + lead);
    p.y = groundY(p) + ANKLE_UP_FT * k;
    return p;
  };
  // Grid search over trunk flexion (1 deg) and lead-foot placement (0.1 ft). Cost: the lead-leg
  // length error (ft, i.e. knee bend off ~45 deg) plus quadratic pulls towards a typical flexion
  // and stride. A long extension cannot have both on a 6 ft 3 in body; quadratics split the
  // difference (linear costs pushed one or the other to its limit).
  let body = null;
  for (let d = 0; d <= MAX_TRUNK_FLEX_DEG; d++) {
    const t = trunk(d * DEG);
    const hip = t.at(`${G}_hip`);
    for (let lead = FRONT_FOOT_LEAD_FT[0] * k; lead <= FRONT_FOOT_LEAD_FT[1] * k + 1e-9; lead += 0.1) {
      const legErr = Math.abs(hip.distanceTo(frontAnkle(hip, lead)) - FRONT_LEG_REACH * LEG_FT * k);
      const stride = PIVOT_FOOT_Y_FT + hip.z + lead;
      const cost = legErr + 0.05 * (((d - TYPICAL_TRUNK_FLEX_DEG) / 10) ** 2 + ((stride - TYPICAL_STRIDE_FT * k) / 0.7) ** 2);
      if (!body || cost < body.cost) body = { ...t, lead, legErr, cost };
    }
  }
  const { q, at } = body;

  const p = REST.map(() => null);
  for (const n of ["pelvis", "spine1", "spine2", "spine3", "neck", "left_hip", "right_hip",
    "left_collar", "right_collar", "left_shoulder", "right_shoulder"]) p[J[n]] = at(n);
  // Head stays up, eyes on the target: it keeps less of the trunk's flexion and tilt.
  const qHead = new THREE.Quaternion().setFromAxisAngle(V(0, 0, 1), s * tilt * 0.5)
    .multiply(new THREE.Quaternion().setFromAxisAngle(V(1, 0, 0), body.flex * 0.4));
  p[J.head] = rest("head").sub(rest("neck")).applyQuaternion(qHead).add(p[J.neck]);

  // Throwing arm: elbow bent behind the shoulder-hand line (towards the rubber, a little down).
  const arm = twoBone(p[J[`${T}_shoulder`]], hand, a, b, V(0, -0.3, -1));
  p[J[`${T}_elbow`]] = arm.mid;
  p[J[`${T}_wrist`]] = arm.mid.clone().add(arm.end.clone().sub(arm.mid).setLength(forearm));
  p[J[`${T}_hand`]] = arm.end;

  // Glove arm tucked in front of the chest, set in the trunk's own frame (x = figure's left).
  const gs = -s;
  const up = V(gs * 0.3, -0.75, 0.6).normalize().applyQuaternion(q);
  const fore = V(-gs * 0.75, 0.2, 0.55).normalize().applyQuaternion(q);
  p[J[`${G}_elbow`]] = p[J[`${G}_shoulder`]].clone().addScaledVector(up, seg(`${G}_shoulder`, `${G}_elbow`));
  p[J[`${G}_wrist`]] = p[J[`${G}_elbow`]].clone().addScaledVector(fore, seg(`${G}_elbow`, `${G}_wrist`));
  p[J[`${G}_hand`]] = p[J[`${G}_wrist`]].clone().addScaledVector(fore, seg(`${G}_wrist`, `${G}_hand`));

  // Lead (glove-side) leg: knee forward.
  const fHip = p[J[`${G}_hip`]];
  const fTarget = frontAnkle(fHip, body.lead);
  const front = twoBone(fHip, fTarget, seg(`${G}_hip`, `${G}_knee`), seg(`${G}_knee`, `${G}_ankle`), V(0, 0.2, 1));
  p[J[`${G}_knee`]] = front.mid;
  p[J[`${G}_ankle`]] = front.end;
  p[J[`${G}_foot`]] = toeTarget(front.end, k);

  // Trailing leg: toe on the ground as far back as the leg reaches, never behind the rubber.
  const bHip = p[J[`${T}_hip`]];
  const L = BACK_LEG_REACH * LEG_FT * k;
  const bTarget = V(bHip.x, groundY(bHip) + BACK_ANKLE_UP_FT * k, 0);
  const dy = bHip.y - bTarget.y;
  const backGrounded = dy < L;
  if (backGrounded) {
    bTarget.z = Math.max(bHip.z - Math.sqrt(L * L - dy * dy), -PIVOT_FOOT_Y_FT);
    bTarget.y = groundY(bTarget) + BACK_ANKLE_UP_FT * k;
  } else bTarget.copy(bHip).addScaledVector(V(0, -1, -0.5).normalize(), L); // hip too high: hangs
  const back = twoBone(bHip, bTarget, seg(`${T}_hip`, `${T}_knee`), seg(`${T}_knee`, `${T}_ankle`), V(0, -1, 0.5));
  p[J[`${T}_knee`]] = back.mid;
  p[J[`${T}_ankle`]] = back.end;
  p[J[`${T}_foot`]] = toeTarget(back.end, k);

  return {
    targets: p,
    flexDeg: body.flex / DEG,
    frontPlanted: front.reached,
    kneeAsWanted: body.legErr < 0.05,
    backGrounded,
    strideFt: PIVOT_FOOT_Y_FT + front.end.z, // pivot foot to lead ankle, along the target line
  };
}

const dirTo = (pts, i, c) => pts[c].clone().sub(pts[i]).normalize();
// Bones with several children: [up child, left child, right child]. Their orientation is taken
// from "up" (along the spine) and "across" (right child -> left child), not from two child
// directions: a video skeleton's hips sit almost straight under its pelvis, and fitting the rig's
// wide-set hip directions to that tilted the whole pelvis ~50 deg. For a rigid pose (solvePose)
// both give the same rotation.
const MULTI = { [J.pelvis]: [J.spine1, J.left_hip, J.right_hip], [J.spine3]: [J.neck, J.left_collar, J.right_collar] };
/** Orientation of joint i: x across (right -> left child), y up (towards the spine child). */
function frame(pts, i) {
  const [up, l, r] = MULTI[i];
  const x = pts[l].clone().sub(pts[r]).normalize();
  const y = dirTo(pts, i, up);
  y.addScaledVector(x, -y.dot(x)).normalize();
  return new THREE.Quaternion().setFromRotationMatrix(new THREE.Matrix4().makeBasis(x, y, x.clone().cross(y)));
}

/**
 * Turn joint positions into bone rotations (rest rotations are identity, so rest offsets are in
 * world axes). A bone with several children (pelvis, spine3) takes the rotation mapping its rest
 * frame onto the target frame; a bone with one child takes the smallest swing, relative to its
 * parent, that points it at the child (no added twist); a leaf follows its parent. Bone lengths
 * are fixed: only directions are taken from `pts` (plus the pelvis position), so positions that
 * are not length-consistent (an averaged video delivery) bend the rig without stretching it.
 * `scale` is the figure's (Group) scale: bone offsets are in rest units, `pts` in world feet.
 */
function setBones(bones, pts, scale = 1) {
  const world = [];
  JOINTS.forEach((j, i) => {
    const parent = j.parent < 0 ? new THREE.Quaternion() : world[j.parent];
    const k = CHILDREN[i];
    let q;
    if (k.length === 0) q = parent.clone();
    else if (k.length === 1) {
      const from = REST[k[0]].clone().sub(REST[i]).normalize().applyQuaternion(parent);
      q = new THREE.Quaternion().setFromUnitVectors(from, dirTo(pts, i, k[0])).multiply(parent);
    } else q = frame(pts, i).multiply(frame(REST, i).invert());
    world[i] = q;
    bones[i].quaternion.copy(parent.clone().invert().multiply(q));
  });
  bones[0].position.copy(pts[0]).divideScalar(scale);
}

/**
 * Pose the figure at ball release: on the mound, the throwing hand on `release` (Statcast ft) and
 * the arm at `armAngleDeg` (Statcast arm_angle). Remembers the arm angle in userData.armAngleDeg.
 */
export function poseAtRelease(pitcher, release, armAngleDeg) {
  const { bones, throws, scale } = pitcher.userData;
  setBones(bones, solvePose(throws, release, armAngleDeg, scale).targets, scale);
  pitcher.userData.armAngleDeg = armAngleDeg;
  pitcher.updateMatrixWorld(true);
}

// Delivery playback (M5). The hand blends onto this pitch's release over the last 0.15 s before
// it, and back onto the delivery's own path by 0.12 s after: long enough not to snap, short
// enough that the windup is the video's own motion.
const BLEND_IN_S = 0.15;
const BLEND_OUT_S = 0.12;
const smoothstep = (x) => (x <= 0 ? 0 : x >= 1 ? 1 : x * x * (3 - 2 * x)); // zero slope at both ends
/** IK blend weight at t s from release: 0 -> 1 over [-0.15, 0], 1 -> 0 over [0, 0.12]. */
const blendWeight = (t) => (t <= 0 ? smoothstep((t + BLEND_IN_S) / BLEND_IN_S) : 1 - smoothstep(t / BLEND_OUT_S));

/**
 * Cubic interpolation of a per-frame array at t s from release (ends held): uniform Catmull-Rom.
 * Why cubic: linear interpolation is a chain of straight segments, so the velocity jumps at every
 * 60 fps frame, which shows as a stutter in slow motion (1/4x, 1/10x). Catmull-Rom (Catmull & Rom
 * 1974) takes each frame's tangent from its two neighbours, (p[i+1] - p[i-1]) / 2, and joins cubic
 * pieces with matching tangents, so velocity is continuous. Uniform (not centripetal) because the
 * samples are evenly spaced in time, which is the parameter we interpolate in.
 * `mix(frames, weights)` combines frames with weights that sum to 1.
 */
function atTime(delivery, frames, t, mix) {
  const n = frames.length;
  const f = Math.min(n - 1, Math.max(0, delivery.release_index + t * delivery.fps));
  const i = Math.min(n - 2, Math.floor(f));
  const u = f - i, u2 = u * u, u3 = u2 * u;
  const at = (k) => frames[Math.min(n - 1, Math.max(0, k))]; // repeat the end frames
  return mix([at(i - 1), at(i), at(i + 1), at(i + 2)],
    [(-u3 + 2 * u2 - u) / 2, (3 * u3 - 5 * u2 + 2) / 2, (-3 * u3 + 4 * u2 + u) / 2, (u3 - u2) / 2]);
}
/** The delivery's 24 joints (scene ft) at t s from release. */
const deliveryFrame = (delivery, t) => atTime(delivery, delivery.joints_ft, t,
  (fr, w) => fr[0].map((_, j) => fr.reduce((s, f, k) => s.addScaledVector(toScene(f[j]), w[k]), new THREE.Vector3())));

/** World positions of all joints as the bones currently put them (forward kinematics). */
function jointPositions(figure) {
  figure.updateMatrixWorld(true);
  return figure.userData.bones.map((b) => b.getWorldPosition(new THREE.Vector3()));
}

/**
 * Where a foot goes: its toe (ball of the foot) on the delivery's toe, kept above the mound, and
 * the ankle one rig foot-length behind it, raised above the toe as much as the delivery's ankle
 * is (heel lift). Pinning the toe, not the ankle, is what keeps a planted foot planted: the video
 * skeleton's foot is half as long as the rig's, so following its direction would drag the toe.
 */
function footTarget(src, sd, k) {
  const sAnkle = src[J[`${sd}_ankle`]], toe = src[J[`${sd}_foot`]].clone();
  toe.y = Math.max(toe.y, groundY(toe) + 0.06 * k);
  const len = FOOT_FT * k;
  const back = V(sAnkle.x - toe.x, 0, sAnkle.z - toe.z);
  if (back.lengthSq() < 1e-8) back.set(0, 0, -1); // foot straight down: any heading
  const rise = Math.min(Math.max(sAnkle.y - toe.y, ANKLE_UP_FT * k - 0.06 * k), 0.95 * len);
  const ankle = toe.clone().addScaledVector(back.normalize(), Math.sqrt(len * len - rise * rise));
  ankle.y += rise;
  return { ankle, toe };
}

/**
 * How far (ft) to lower the pelvis at each delivery frame so the rig's legs reach the delivery's
 * ankles. The video skeleton's legs (and its pelvis-to-hip drop) are ~0.3 ft longer than the
 * stylised rig's, so with the pelvis where the video has it the planted pivot foot would float
 * and slide. Smoothed (max, then mean, over +-0.1 s) so the body does not bob. Cached per
 * delivery on the figure: computed once, 187 frames of FK.
 */
function pelvisDrops(figure, delivery) {
  const ud = figure.userData;
  if (ud.dropsFor === delivery) return ud.drops;
  const k = ud.scale;
  const R = 0.9995 * LEG_FT * k; // straight leg (twoBone straightens at 0.9999)
  const raw = delivery.joints_ft.map((fr) => {
    const src = fr.map(toScene);
    setBones(ud.bones, src, k);
    const p = jointPositions(figure);
    return Math.max(0, ...["left", "right"].map((sd) => {
      const d = p[J[`${sd}_hip`]].clone().sub(footTarget(src, sd, k).ankle);
      const h = Math.hypot(d.x, d.z);
      return h < R ? d.y - Math.sqrt(R * R - h * h) : d.y; // drop that brings the ankle into reach
    }));
  });
  const W = Math.round(0.1 * delivery.fps);
  const around = (a, i) => a.slice(Math.max(0, i - W), i + W + 1);
  const high = raw.map((_, i) => Math.max(...around(raw, i)));
  ud.drops = high.map((_, i) => around(high, i).reduce((s, v) => s + v) / around(high, i).length);
  ud.dropsFor = delivery;
  return ud.drops;
}

/**
 * Pose the figure from a recovered delivery (M5 JSON) at t s from release (t = 0).
 *
 * 1. The rig's fixed-length bones point where the delivery's segments point. Averaging many
 *    clips shrinks fast-moving segments (the delivery's lengths vary over time), so positions are
 *    not copied: bones follow directions, and the rig never stretches.
 * 2. Legs: following directions with different lengths would slide the feet, so the pelvis is
 *    lowered until the legs reach (pelvisDrops) and each leg is re-solved by two-bone IK so the
 *    toe lands on the delivery's toe (footTarget; planted in the video), the knee bending the
 *    way the delivery's does. Ankle and toe are kept above the mound.
 * 3. Throwing arm: blended (weight blendWeight(t)) from the delivery's arm to the two-bone IK
 *    arm whose hand is on this pitch's `release`. Each bone's direction is slerped, so lengths
 *    hold and the hand glides onto the release instead of snapping. A null release: no blend.
 *
 * @param release Statcast [x, y, z] ft, or null
 * @param armAngleDeg kept in userData like poseAtRelease (the arm itself follows the delivery)
 */
export function poseAtTime(figure, delivery, t, release, armAngleDeg) {
  const { bones, throws, scale: k } = figure.userData;
  const drop = atTime(delivery, pelvisDrops(figure, delivery), t, (v, w) => v.reduce((s, x, k) => s + x * w[k], 0));
  const src = deliveryFrame(delivery, t);
  setBones(bones, src, k);
  const p = jointPositions(figure).map((v) => v.setY(v.y - drop));

  for (const sd of ["left", "right"]) {
    const [hip, knee, ankle, foot] = ["hip", "knee", "ankle", "foot"].map((n) => J[`${sd}_${n}`]);
    const target = footTarget(src, sd, k);
    // Pole: the delivery's knee off its hip-ankle midpoint. When the leg is nearly straight this
    // is short and noisy, but then the knee sits on the line whatever its direction.
    const pole = src[knee].clone().sub(src[hip].clone().add(src[ankle]).multiplyScalar(0.5));
    const leg = twoBone(p[hip], target.ankle, k * segFt(`${sd}_hip`, `${sd}_knee`), k * segFt(`${sd}_knee`, `${sd}_ankle`), pole);
    p[knee] = leg.mid;
    p[ankle] = leg.end;
    // Toe on its target (exactly, when the leg reached), lifted if it would dig into the mound.
    const len = FOOT_FT * k;
    const dir = target.toe.clone().sub(p[ankle]).setLength(len);
    const dy = groundY(p[ankle]) + 0.06 * k - p[ankle].y; // lowest allowed toe drop (negative)
    if (dir.y < dy) {
      const flat = Math.hypot(dir.x, dir.z) || 1;
      const h = Math.sqrt(Math.max(len * len - dy * dy, 0));
      dir.set((dir.x / flat) * h, dy, (dir.z / flat) * h);
    }
    p[foot] = p[ankle].clone().add(dir);
  }

  const w = release ? blendWeight(t) : 0;
  if (w > 0) {
    const T = throws === "L" ? "left" : "right";
    const [sh, el, wr, hd] = ["shoulder", "elbow", "wrist", "hand"].map((n) => J[`${T}_${n}`]);
    const a = k * segFt(`${T}_shoulder`, `${T}_elbow`);
    const forearm = k * segFt(`${T}_elbow`, `${T}_wrist`);
    const palm = k * segFt(`${T}_wrist`, `${T}_hand`);
    // Elbow bends the way the delivery's does at this instant (pole = its elbow off the shoulder).
    const ik = twoBone(p[sh], toScene(release), a, forearm + palm, p[el].clone().sub(p[sh]));
    const ikWrist = ik.mid.clone().add(ik.end.clone().sub(ik.mid).setLength(forearm));
    const to = [p[sh], ik.mid, ikWrist, ik.end];
    const chain = [[sh, el, a], [el, wr, forearm], [wr, hd, palm]];
    const from = chain.map(([i, c]) => p[c].clone().sub(p[i]).normalize()); // the delivery's arm
    chain.forEach(([i, c, len], n) => {
      const goal = to[n + 1].clone().sub(to[n]).normalize();
      const q = new THREE.Quaternion().setFromUnitVectors(from[n], goal);
      p[c] = p[i].clone().addScaledVector(from[n].clone().applyQuaternion(new THREE.Quaternion().slerp(q, w)), len);
    });
  }
  setBones(bones, p, k);
  figure.userData.armAngleDeg = armAngleDeg;
  figure.updateMatrixWorld(true);
}

/** World position (scene ft) of the throwing hand's knuckles, where the ball sits. */
export function handPosition(figure, throws = figure.userData.throws) {
  figure.updateMatrixWorld(true);
  return figure.userData.bones[J[`${throws === "L" ? "left" : "right"}_hand`]].getWorldPosition(new THREE.Vector3());
}

// Materials: one park-style toon material per body part (look.js), all made here.
const COLORS = {
  jersey: PALETTE.jersey, sleeve: PALETTE.navy, pants: PALETTE.pants, skin: PALETTE.skin,
  cap: PALETTE.navy, glove: PALETTE.glove, shoe: PALETTE.shoe,
};
const makeMaterials = () => Object.fromEntries(
  Object.entries(COLORS).map(([k, c]) => [k, makeToonMaterial(c)]));

/** Rigid skinning: each vertex follows one bone, chosen from its rest position. */
function skin(geo, boneOf) {
  const pos = geo.attributes.position;
  const idx = new Uint16Array(pos.count * 4);
  const wt = new Float32Array(pos.count * 4);
  const v = new THREE.Vector3();
  for (let i = 0; i < pos.count; i++) {
    idx[i * 4] = boneOf(v.fromBufferAttribute(pos, i));
    wt[i * 4] = 1;
  }
  geo.setAttribute("skinIndex", new THREE.Uint16BufferAttribute(idx, 4));
  geo.setAttribute("skinWeight", new THREE.Float32BufferAttribute(wt, 4));
  return geo;
}

/** Capsule of radius r between two rest joints (ends extended by `ext` ft), on joint a's bone. */
function limb(a, b, r, ext = 0) {
  const pa = REST[J[a]], pb = REST[J[b]];
  const d = pb.clone().sub(pa);
  const g = new THREE.CapsuleGeometry(r, Math.max(d.length() + 2 * ext - 2 * r, 0.01), 6, 14);
  g.applyQuaternion(new THREE.Quaternion().setFromUnitVectors(V(0, 1, 0), d.clone().normalize()));
  g.translate(...pa.clone().add(pb).multiplyScalar(0.5).toArray());
  return skin(g, () => J[a]);
}

/** A lathe (radius, height) profile, squashed to an oval body cross-section. */
function lathe(profile, sx, sz) {
  const g = new THREE.LatheGeometry(profile.map(([r, y]) => new THREE.Vector2(r, y)), 28);
  g.scale(sx, 1, sz);
  return g;
}

/** Stylised body: lathe torso and hips, capsule limbs, head with a cap, glove on the glove hand. */
function bodyParts(throws) {
  const G = throws === "L" ? "right" : "left";
  // The torso follows the spine bone whose segment each vertex sits in (heights from the rest pose).
  const spineAt = (v) => (v.y < REST[J.spine1].y ? J.pelvis
    : v.y < REST[J.spine2].y ? J.spine1 : v.y < REST[J.spine3].y ? J.spine2 : J.spine3);
  const parts = [
    ["jersey", skin(lathe([[0, 3.4], [0.44, 3.42], [0.42, 3.8], [0.44, 4.2], [0.5, 4.6], [0.54, 4.95],
      [0.46, 5.15], [0.24, 5.3], [0, 5.32]], 1.3, 0.72), spineAt)],
    ["pants", skin(lathe([[0, 2.85], [0.36, 2.9], [0.47, 3.15], [0.46, 3.5], [0, 3.55]], 1.25, 0.75), () => J.pelvis)],
    ["skin", limb("neck", "head", 0.17)],
    ["skin", skin(new THREE.SphereGeometry(0.36, 24, 16).translate(0, 5.86, 0.04), () => J.head)],
    // Cap crown and a brim pointing forward.
    ["cap", skin(new THREE.SphereGeometry(0.375, 24, 8, 0, 2 * Math.PI, 0, Math.PI / 2)
      .scale(1, 0.85, 1).translate(0, 5.92, 0.03), () => J.head)],
    ["cap", skin(new THREE.CylinderGeometry(0.26, 0.26, 0.03, 20).scale(1, 1, 0.8)
      .translate(0, 5.94, 0.42), () => J.head)],
  ];
  for (const sd of ["left", "right"]) {
    parts.push(
      ["sleeve", limb(`${sd}_shoulder`, `${sd}_elbow`, 0.19, 0.12)],
      ["skin", limb(`${sd}_elbow`, `${sd}_wrist`, 0.14, 0.05)],
      ["pants", limb(`${sd}_hip`, `${sd}_knee`, 0.27, 0.1)],
      ["pants", limb(`${sd}_knee`, `${sd}_ankle`, 0.19, 0.05)],
      ["shoe", limb(`${sd}_ankle`, `${sd}_foot`, 0.14, 0.12)],
      sd === G
        ? ["glove", skin(new THREE.SphereGeometry(0.3, 18, 12).scale(1, 1.15, 0.7)
          .translate(...REST[J[`${sd}_hand`]].toArray()), () => J[`${sd}_wrist`])]
        : ["skin", limb(`${sd}_wrist`, `${sd}_hand`, 0.11, 0.06)],
    );
  }
  return parts;
}

/**
 * The pitcher: a Group holding one SkinnedMesh on the SMPL/Mixamo skeleton (skeleton.js),
 * posed at a typical release until poseAtRelease is called with a real pitch. The rest skeleton
 * is 75 in tall; the figure is scaled by heightIn / 75.
 * userData: {throws, bones (JOINTS order), scale, armAngleDeg}.
 */
export function makePitcher(throws, heightIn = 75) {
  const { bones, skeleton } = buildSkeleton();
  const parts = bodyParts(throws);
  const mats = makeMaterials();
  const mesh = new THREE.SkinnedMesh(mergeGeometries(parts.map((p) => p[1]), true), parts.map((p) => mats[p[0]]));
  mesh.add(bones[0]);
  mesh.bind(skeleton); // at rest: T-pose is the bind pose
  mesh.frustumCulled = false; // bounds are from the rest pose at the origin, not on the mound
  mesh.castShadow = mesh.receiveShadow = true;
  const g = new THREE.Group();
  g.add(mesh);
  const scale = heightIn / 75;
  g.scale.setScalar(scale); // uniform: bone rotations are unchanged, every length scales
  g.userData = { throws, bones, scale };
  poseAtRelease(g, [throws === "L" ? 2 : -2, 54.5, 5.6], 35);
  return g;
}
