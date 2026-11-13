import { test } from "node:test";
import assert from "node:assert/strict";
import { Vector3 } from "three";
import { JOINTS } from "../src/skeleton.js";
import { makePitcher, poseAtRelease } from "../src/pitcher.js";
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
