import { test } from "node:test";
import assert from "node:assert/strict";
import { Vector3, Quaternion, Line3 as THREE_Line } from "three";
import { readFileSync } from "node:fs";
import { JOINTS, J } from "../src/skeleton.js";
import { makePitcher, poseAtRelease, poseAtTime, handPosition } from "../src/pitcher.js";
import { toScene } from "../src/coords.js";

// SMPL's kinematic tree (Loper et al. 2015, SMPL model files): joint order and parents.
const SMPL_ORDER = [
  "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee", "spine2",
  "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot", "neck", "left_collar",
  "right_collar", "head", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
  "left_wrist", "right_wrist", "left_hand", "right_hand",
];
const SMPL_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21];

test("joint list: SMPL's 24 joints in SMPL order, each mapped to a unique Mixamo name", () => {
  assert.equal(JOINTS.length, 24);
  assert.deepEqual(JOINTS.map((j) => j.smpl), SMPL_ORDER);
  assert.deepEqual(JOINTS.map((j) => j.parent), SMPL_PARENTS);
  for (const j of JOINTS) assert.match(j.mixamo, /^mixamorig:[A-Z]\w+$/);
  assert.equal(new Set(JOINTS.map((j) => j.mixamo)).size, 24);
  // Anatomical sides survive the renaming: SMPL left_* is Mixamo Left*.
  for (const j of JOINTS) {
    if (j.smpl.startsWith("left_")) assert.match(j.mixamo, /:Left/);
    if (j.smpl.startsWith("right_")) assert.match(j.mixamo, /:Right/);
  }
});

const handName = (side) => JOINTS.find((j) => j.smpl === `${side}_hand`).mixamo;
const shoulderName = (side) => JOINTS.find((j) => j.smpl === `${side}_shoulder`).mixamo;
const worldPos = (fig, name) => fig.getObjectByName(name).getWorldPosition(new Vector3());

for (const [throws, side, release] of [["R", "right", [-2.4, 54.3, 5.7]], ["L", "left", [2.9, 54.2, 5.0]]]) {
  test(`hand at release (${throws}HP): the ${side} hand sits on Statcast's release point`, () => {
    const fig = makePitcher(throws);
    poseAtRelease(fig, release, 30);
    fig.updateMatrixWorld(true);
    const d = worldPos(fig, handName(side)).distanceTo(toScene(release));
    assert.ok(d < 0.05, `hand ${d.toFixed(3)} ft from release`);
  });
}

test("arm angle: shoulder-to-hand elevation above horizontal equals arm_angle", () => {
  for (const throws of ["R", "L"]) {
    const side = throws === "R" ? "right" : "left";
    const release = [throws === "R" ? -2.2 : 2.2, 54.5, 5.6];
    for (const deg of [10, 40, 70]) {
      const fig = makePitcher(throws);
      poseAtRelease(fig, release, deg);
      fig.updateMatrixWorld(true);
      const v = worldPos(fig, handName(side)).sub(worldPos(fig, shoulderName(side)));
      const elev = (Math.atan2(v.y, Math.hypot(v.x, v.z)) * 180) / Math.PI;
      assert.ok(Math.abs(elev - deg) < 2, `${throws}HP ${deg}: got ${elev.toFixed(2)}`);
      // The hand is out on the throwing side, past the shoulder (Statcast x: RHP < 0).
      assert.ok(throws === "R" ? v.x < 0 : v.x > 0);
    }
  }
});

// --- Delivery playback (M5). SYNTHETIC fixture (not from video): the 75 in rest pose on the
// mound, right arm swinging overhand, release at frame 60 of 60 fps (t = 0). See its "note".
const DELIVERY = JSON.parse(readFileSync(new URL("./fixtures/delivery-synthetic.json", import.meta.url)));
const fixtureJoint = (t, name) => toScene(DELIVERY.joints_ft[DELIVERY.release_index + t * DELIVERY.fps][J[name]]);
const add = ([x, y, z], [a, b, c], s = 1) => [x + s * a, y + s * b, z + s * c];
const ownHand0 = DELIVERY.joints_ft[DELIVERY.release_index][J.right_hand]; // Statcast ft

test("delivery: hand reaches the release point at t = 0", () => {
  const fig = makePitcher("R");
  const release = add(ownHand0, [0.12, -0.1, -0.1]);
  poseAtTime(fig, DELIVERY, 0, release, 45);
  const d = handPosition(fig, "R").distanceTo(toScene(release));
  assert.ok(d < 0.01, `hand ${d.toFixed(4)} ft from release`);
});

test("delivery: a far release is reached without a snap", () => {
  const fig = makePitcher("R");
  const release = add(ownHand0, [0.6, 0, -0.8], 0.8); // 0.8 ft from the delivery's own hand
  let prev = null, maxStep = 0, maxElbow = 0;
  for (let i = -72; i <= 24; i++) { // t = -0.3 .. 0.1 s in 1/240 s steps
    poseAtTime(fig, DELIVERY, i / 240, release, 45);
    const h = handPosition(fig, "R");
    if (prev) maxStep = Math.max(maxStep, h.distanceTo(prev));
    prev = h;
    const e = worldPos(fig, "mixamorig:RightForeArm");
    const elbow = worldPos(fig, "mixamorig:RightArm").sub(e).angleTo(worldPos(fig, "mixamorig:RightHand").sub(e));
    maxElbow = Math.max(maxElbow, (elbow * 180) / Math.PI);
  }
  assert.ok(maxStep <= 0.25, `largest hand step ${maxStep.toFixed(3)} ft`);
  assert.ok(maxElbow < 178, `elbow reached ${maxElbow.toFixed(1)} deg`);
  poseAtTime(fig, DELIVERY, 0, release, 45);
  assert.ok(handPosition(fig, "R").distanceTo(toScene(release)) < 0.01);
});

test("delivery: before the blend the delivery is untouched", () => {
  const fig = makePitcher("R");
  poseAtTime(fig, DELIVERY, -0.5, add(ownHand0, [0.5, 0, -0.5]), 45);
  const d = handPosition(fig, "R").distanceTo(fixtureJoint(-0.5, "right_hand"));
  assert.ok(d < 1e-6, `hand ${d} ft off the fixture frame`);
});

test("height scales the figure, and the hand still reaches the release", () => {
  for (const heightIn of [70, 80]) {
    const fig = makePitcher("R", heightIn);
    // Standing height: top of the rest-pose (T-pose, floor at 0) mesh, times the figure's scale.
    const mesh = fig.children[0];
    mesh.geometry.computeBoundingBox();
    const top = mesh.geometry.boundingBox.max.y * fig.getWorldScale(new Vector3()).y;
    assert.ok(Math.abs(top - heightIn / 12) < 0.1, `${heightIn} in: head top ${top.toFixed(3)} ft`);
    // A delivery plays on its own pitcher (same height). Here the 75 in fixture drives a 70 / 80
    // in body, whose shoulder sits lower / higher, so the release is taken near where this body's
    // own (unblended) hand is at t = 0, then moved 0.2 ft.
    poseAtTime(fig, DELIVERY, 0, null, 45);
    const h = handPosition(fig, "R");
    const release = add([h.x, -h.z, h.y], [0.12, -0.1, -0.13]); // scene -> Statcast, + 0.2 ft
    poseAtTime(fig, DELIVERY, 0, release, 45);
    const d = handPosition(fig, "R").distanceTo(toScene(release));
    assert.ok(d < 0.01, `${heightIn} in: hand ${d.toFixed(4)} ft from release`);
    // The static pose scales too (pitchers without a delivery).
    poseAtRelease(fig, [-2.2, 54.5, 5.6], 35);
    assert.ok(handPosition(fig, "R").distanceTo(toScene([-2.2, 54.5, 5.6])) < 0.05);
  }
});

test("delivery: hand velocity is continuous across frame boundaries (slow motion stays smooth)", () => {
  // SYNTHETIC: the fixture with the whole body sliding sideways with constant acceleration,
  // x += A t^2 (Statcast ft). Linear interpolation between 60 fps frames turns that into a chain of
  // straight segments, so the velocity jumps by 2A/fps at every frame; a cubic (Catmull-Rom) does not.
  const A = 2; // ft/s^2
  const fps = DELIVERY.fps;
  const moving = { ...DELIVERY, joints_ft: DELIVERY.joints_ft.map((fr, i) => {
    const t = (i - DELIVERY.release_index) / fps;
    return fr.map(([x, y, z]) => [x + A * t * t, y, z]);
  }) };
  const fig = makePitcher("R");
  const hand = (t) => { poseAtTime(fig, moving, t, null, 45); return handPosition(fig, "R"); };
  const h = 1e-4;
  for (const tb of [-0.5, -0.25, -0.1]) { // frame boundaries (whole frames at 60 fps)
    const before = hand(tb).sub(hand(tb - h)).divideScalar(h);
    const after = hand(tb + h).sub(hand(tb)).divideScalar(h);
    const jump = after.distanceTo(before);
    assert.ok(jump < 0.25 * (2 * A) / fps, `t=${tb}: velocity jumps ${jump.toFixed(4)} ft/s at the frame boundary (linear: ~${((2 * A) / fps).toFixed(3)})`);
  }
});

// --- The figure's meshes (figure.js). Skinned vertex positions come from three's own skinning
// maths (SkinnedMesh.applyBoneTransform), so these check what is drawn.
const meshOf = (fig) => fig.children[0];
/** World positions of the vertices whose main bone is `bone` (all vertices if null), in group `mat`. */
function skinnedVerts(fig, { bone = null, mat = null } = {}) {
  const mesh = meshOf(fig);
  fig.updateMatrixWorld(true);
  const g = mesh.geometry, si = g.attributes.skinIndex, ix = g.index.array;
  const groups = mat == null ? [{ start: 0, count: ix.length }]
    : g.groups.filter((gr) => mesh.material[gr.materialIndex].name === mat);
  const seen = new Set(), out = [];
  for (const gr of groups) for (let n = gr.start; n < gr.start + gr.count; n++) {
    const i = ix[n];
    if (seen.has(i) || (bone != null && si.getX(i) !== bone)) continue;
    seen.add(i);
    out.push({ bone: si.getX(i), p: mesh.applyBoneTransform(i, new Vector3().fromBufferAttribute(g.attributes.position, i)).applyMatrix4(mesh.matrixWorld) });
  }
  return out;
}

for (const [throws, G] of [["R", "left"], ["L", "right"]]) {
  test(`glove (${throws}HP): the mitt is bound to the ${G} (glove-side) hand and sits on it`, () => {
    const fig = makePitcher(throws);
    const verts = skinnedVerts(fig, { mat: "glove" });
    assert.ok(verts.length > 100, "a mitt with some shape");
    for (const { bone } of verts) assert.equal(bone, J[`${G}_wrist`]);
    const c = verts.reduce((s, v) => s.add(v.p), new Vector3()).divideScalar(verts.length);
    const hand = worldPos(fig, handName(G));
    assert.ok(c.distanceTo(hand) < 0.5, `mitt centre ${c.distanceTo(hand).toFixed(2)} ft from the ${G} knuckles`);
  });
}

test("ball grip: through the windup the ball at handPosition sits between fingers and thumb", () => {
  const BALL_FT = 1.45 / 12;
  for (const [throws, T] of [["R", "right"], ["L", "left"]]) {
    const fig = makePitcher(throws);
    for (const t of [-2, -1.2, -0.6, -0.2, 0]) {
      if (throws === "R") poseAtTime(fig, DELIVERY, t, null, 45);
      const ball = handPosition(fig, throws);
      const hand = skinnedVerts(fig, { bone: J[`${T}_wrist`] }).map((v) => v.p);
      const d = hand.map((p) => p.distanceTo(ball));
      // Nothing of the hand inside the ball (a little overlap allowed for the touching pads)...
      assert.ok(Math.min(...d) > BALL_FT - 0.02, `${throws} t=${t}: hand ${Math.min(...d).toFixed(3)} ft from the ball centre`);
      // ...and the hand touches it on opposite sides (fingers over, thumb under): hand points
      // within a finger's thickness of the surface, in directions more than 120 deg apart.
      const touch = hand.filter((p, i) => d[i] < BALL_FT + 0.03).map((p) => p.clone().sub(ball).normalize());
      const spread = Math.max(...touch.flatMap((a) => touch.map((b) => a.angleTo(b))));
      assert.ok(spread > (120 * Math.PI) / 180, `${throws} t=${t}: grip spans ${((spread * 180) / Math.PI).toFixed(0)} deg`);
      if (throws === "L") break; // the fixture is a right-hander's: the left grip is checked at rest
    }
  }
});

test("no part detaches: every vertex stays within reach of its bone through the delivery", () => {
  for (const heightIn of [75, 80]) {
    const fig = makePitcher("R", heightIn);
    const k = heightIn / 75;
    const bones = fig.userData.bones;
    const kids = JOINTS.map((_, i) => JOINTS.findIndex((j) => j.parent === i));
    for (const t of [-2.5, -1.5, -0.9, -0.4, -0.1, 0, 0.3]) {
      poseAtTime(fig, DELIVERY, t, add(ownHand0, [0.12, -0.1, -0.1]), 45);
      const at = bones.map((b) => b.getWorldPosition(new Vector3()));
      for (const { bone, p } of skinnedVerts(fig)) {
        // Distance to the bone's segment (joint -> first child; a leaf is a point). The farthest
        // legitimate parts: the mitt's tip ~1 ft past the wrist, the shoe's toe past the ankle.
        const a = at[bone], b = kids[bone] >= 0 ? at[kids[bone]] : a;
        const d = a === b ? a.distanceTo(p) : new THREE_Line(a, b).closestPointToPoint(p, true, new Vector3()).distanceTo(p);
        assert.ok(d < 1.1 * k, `t=${t}: a vertex on ${JOINTS[bone].smpl} is ${d.toFixed(2)} ft off its bone`);
      }
    }
  }
});

test("steady arm roll: arm bones turn no faster than the arm swings (no added twist)", () => {
  // The fixture's arm swings at 200 deg/s, i.e. 0.83 deg per 1/240 s sample. The smallest swing
  // from the parent adds roll about the bone's own axis as the arm passes overhead (1.36 deg per
  // sample with it); rolled by the elbow's hinge plane the bones turn only as the arm does.
  // Windup bound 1.0 deg (the true 0.83 + interpolation slack); after it, a loose 3 deg.
  const fig = makePitcher("R");
  const arm = ["shoulder", "elbow", "wrist", "hand"].flatMap((n) => [`left_${n}`, `right_${n}`]);
  const quat = () => arm.map((n) => fig.getObjectByName(JOINTS[J[n]].mixamo).getWorldQuaternion(new Quaternion()));
  let prev = null;
  for (let i = -240; i <= 72; i++) {
    poseAtTime(fig, DELIVERY, i / 240, null, 45);
    const q = quat();
    if (prev) q.forEach((x, b) => {
      const deg = (x.angleTo(prev[b]) * 180) / Math.PI, bound = i < -72 ? 1.0 : 3;
      assert.ok(deg <= bound, `t=${(i / 240).toFixed(3)}: ${arm[b]} turned ${deg.toFixed(2)} deg in 1/240 s`);
    });
    prev = q;
  }
});

test("no bob: frame-to-frame jitter in the video's height does not reach the pelvis", () => {
  // SYNTHETIC: the fixture with the whole body jittering up and down 0.08 ft on alternate frames
  // (what blur at release does to a video skeleton). High-frequency part of the pelvis height =
  // height minus its 0.25 s moving average, over the windup.
  const jitter = { ...DELIVERY, joints_ft: DELIVERY.joints_ft.map((fr, i) => fr.map(([x, y, z]) => [x, y, z + (i % 2 ? 0.08 : -0.08)])) };
  const fig = makePitcher("R");
  const ts = [], ys = [];
  for (let i = -240; i <= -72; i++) {
    poseAtTime(fig, jitter, i / 240, null, 45);
    ts.push(i / 240);
    ys.push(fig.userData.bones[0].getWorldPosition(new Vector3()).y);
  }
  const W = 30; // +-0.125 s at 240 samples/s
  let worst = 0;
  for (let i = W; i < ys.length - W; i++) {
    const avg = ys.slice(i - W, i + W + 1).reduce((s, v) => s + v) / (2 * W + 1);
    worst = Math.max(worst, Math.abs(ys[i] - avg));
  }
  assert.ok(worst < 0.01, `pelvis bobs ${worst.toFixed(3)} ft`);
});
