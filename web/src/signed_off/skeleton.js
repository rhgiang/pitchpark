// FROZEN copy of 70c9f22 (see index.js). Only change: imports of shared coords/field/look point up one folder.
import * as THREE from "three";

/**
 * The pitcher's skeleton: SMPL's 24 joints, in SMPL order (Loper et al. 2015, "SMPL: A Skinned
 * Multi-Person Linear Model"), which is what M5's video-to-3D models (4D-Humans, WHAM) output.
 * Each joint also carries the Mixamo bone name a Blender/Mixamo rig would use, so a modelled
 * character can be bound to the same bones later. THIS TABLE IS THE ONE SMPL <-> MIXAMO MAPPING.
 *
 * Notes on the mapping:
 * - SMPL "left"/"right" are the subject's own sides, as are Mixamo's.
 * - SMPL's spine1/2/3 -> Mixamo Spine/Spine1/Spine2; SMPL collar -> Mixamo *Shoulder (clavicle);
 *   SMPL shoulder -> Mixamo *Arm (the upper-arm bone starts at the shoulder joint).
 * - SMPL wrist -> Mixamo *Hand (the hand bone pivots at the wrist). SMPL "hand" is the middle of
 *   the knuckles, which Mixamo has no single bone for: *HandMiddle1 (middle finger base) is the
 *   closest. The ball sits there at release, so that is the joint pinned to the release point.
 * - SMPL foot (ball of the foot) -> Mixamo *ToeBase.
 *
 * Rest pose: T-pose, scene feet (x = the figure's left, y up, z = facing direction), floor at y = 0,
 * for a 6 ft 3 in (75 in) pitcher. Segment lengths roughly follow Drillis & Contini (1966)
 * fractions of height H, trimmed so arm span ~= height:
 *   shoulder joint height 5.10 ft (0.82 H), hip joints 3.12 ft, knees 1.75 ft, ankles 0.28 ft;
 *   thigh 1.37, shin 1.47 (leg 2.84); shoulder joints 1.24 apart; upper arm 1.05, forearm 0.85,
 *   wrist to knuckles 0.28 (shoulder to knuckles 2.18 straight); foot (ankle to ball) 0.52.
 */
export const JOINTS = [
  { smpl: "pelvis", mixamo: "mixamorig:Hips", parent: -1, rest: [0, 3.3, 0] },
  { smpl: "left_hip", mixamo: "mixamorig:LeftUpLeg", parent: 0, rest: [0.3, 3.12, 0] },
  { smpl: "right_hip", mixamo: "mixamorig:RightUpLeg", parent: 0, rest: [-0.3, 3.12, 0] },
  { smpl: "spine1", mixamo: "mixamorig:Spine", parent: 0, rest: [0, 3.65, -0.03] },
  { smpl: "left_knee", mixamo: "mixamorig:LeftLeg", parent: 1, rest: [0.3, 1.75, 0.02] },
  { smpl: "right_knee", mixamo: "mixamorig:RightLeg", parent: 2, rest: [-0.3, 1.75, 0.02] },
  { smpl: "spine2", mixamo: "mixamorig:Spine1", parent: 3, rest: [0, 4.1, 0] },
  { smpl: "left_ankle", mixamo: "mixamorig:LeftFoot", parent: 4, rest: [0.3, 0.28, -0.08] },
  { smpl: "right_ankle", mixamo: "mixamorig:RightFoot", parent: 5, rest: [-0.3, 0.28, -0.08] },
  { smpl: "spine3", mixamo: "mixamorig:Spine2", parent: 6, rest: [0, 4.35, 0.02] },
  { smpl: "left_foot", mixamo: "mixamorig:LeftToeBase", parent: 7, rest: [0.32, 0.06, 0.4] },
  { smpl: "right_foot", mixamo: "mixamorig:RightToeBase", parent: 8, rest: [-0.32, 0.06, 0.4] },
  { smpl: "neck", mixamo: "mixamorig:Neck", parent: 9, rest: [0, 5.3, -0.05] },
  { smpl: "left_collar", mixamo: "mixamorig:LeftShoulder", parent: 9, rest: [0.25, 5.05, 0] },
  { smpl: "right_collar", mixamo: "mixamorig:RightShoulder", parent: 9, rest: [-0.25, 5.05, 0] },
  { smpl: "head", mixamo: "mixamorig:Head", parent: 12, rest: [0, 5.62, 0.02] },
  { smpl: "left_shoulder", mixamo: "mixamorig:LeftArm", parent: 13, rest: [0.62, 5.1, -0.05] },
  { smpl: "right_shoulder", mixamo: "mixamorig:RightArm", parent: 14, rest: [-0.62, 5.1, -0.05] },
  { smpl: "left_elbow", mixamo: "mixamorig:LeftForeArm", parent: 16, rest: [1.67, 5.1, -0.05] },
  { smpl: "right_elbow", mixamo: "mixamorig:RightForeArm", parent: 17, rest: [-1.67, 5.1, -0.05] },
  { smpl: "left_wrist", mixamo: "mixamorig:LeftHand", parent: 18, rest: [2.52, 5.1, -0.05] },
  { smpl: "right_wrist", mixamo: "mixamorig:RightHand", parent: 19, rest: [-2.52, 5.1, -0.05] },
  { smpl: "left_hand", mixamo: "mixamorig:LeftHandMiddle1", parent: 20, rest: [2.8, 5.1, -0.05] },
  { smpl: "right_hand", mixamo: "mixamorig:RightHandMiddle1", parent: 21, rest: [-2.8, 5.1, -0.05] },
];

/** Index of an SMPL joint name in JOINTS (and in the bones array). */
export const J = Object.fromEntries(JOINTS.map((j, i) => [j.smpl, i]));

/**
 * Bones in T-pose (identity rotations, each positioned relative to its parent), named with the
 * Mixamo names, and their Skeleton with bind inverses taken at the rest pose.
 * @returns {{bones: THREE.Bone[], skeleton: THREE.Skeleton}} bones in JOINTS order
 */
export function buildSkeleton() {
  const bones = JOINTS.map((j) => Object.assign(new THREE.Bone(), { name: j.mixamo }));
  JOINTS.forEach((j, i) => {
    bones[i].position.fromArray(j.rest);
    if (j.parent >= 0) {
      bones[i].position.sub(new THREE.Vector3(...JOINTS[j.parent].rest));
      bones[j.parent].add(bones[i]);
    }
  });
  bones[0].updateMatrixWorld(true); // Skeleton() takes the bind inverses from matrixWorld
  return { bones, skeleton: new THREE.Skeleton(bones) };
}
