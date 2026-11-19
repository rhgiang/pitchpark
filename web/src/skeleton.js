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
export function buildSkeleton(rest = JOINTS.map((j) => new THREE.Vector3(...j.rest))) {
  const bones = JOINTS.map((j) => Object.assign(new THREE.Bone(), { name: j.mixamo }));
  JOINTS.forEach((j, i) => {
    bones[i].position.copy(rest[i]);
    if (j.parent >= 0) {
      bones[i].position.sub(rest[j.parent]);
      bones[j.parent].add(bones[i]);
    }
  });
  bones[0].updateMatrixWorld(true); // Skeleton() takes the bind inverses from matrixWorld
  return { bones, skeleton: new THREE.Skeleton(bones) };
}

const REST_V = JOINTS.map((j) => new THREE.Vector3(...j.rest));
const mirrorName = (n) => (n.startsWith("left_") ? "right_" + n.slice(5) : n.startsWith("right_") ? "left_" + n.slice(6) : n);
/** Length (ft) of the bone from joint i to its parent in the default rig (0 for the pelvis). */
const defaultLen = (i) => (JOINTS[i].parent < 0 ? 0 : REST_V[i].distanceTo(REST_V[JOINTS[i].parent]));
/** Standing height (ft) of the default rig: the 75 in rest size. */
export const RIG_HEIGHT_FT = 75 / 12;
const LEN_RATIO = [0.8, 1.25]; // plausible range of a pitcher's bone vs the default rig's

/**
 * One pitcher's own bone lengths (ft, joint -> parent, by SMPL joint name) read off his delivery
 * (M5 JSON, `joints_ft` [T][24][3]). Why median over frames: the lifted skeleton's lengths are
 * nearly constant (it is one body), and a depth-collapsed or blurred frame is an outlier the
 * median ignores. Why averaged left/right: the lift's L/R difference is noise (a person's two
 * arms and legs are the same length to ~1%). Each length is clamped to LEN_RATIO of the default
 * rig's: the lift's pelvis-to-hip and collar bones are not anatomical lengths (hips sit nearly
 * under the pelvis), and a wild value would only distort the mesh.
 * @returns {Record<string, number>}
 */
export function boneLengths(delivery) {
  const med = (a) => { const s = [...a].sort((x, y) => x - y); return s[s.length >> 1]; };
  const own = JOINTS.map((j, i) => (j.parent < 0 ? 0 : med(delivery.joints_ft.map((f) => {
    const [a, b] = [f[i], f[j.parent]];
    return Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
  }))));
  return Object.fromEntries(JOINTS.flatMap((j, i) => {
    if (j.parent < 0) return [];
    const len = (own[i] + own[J[mirrorName(j.smpl)]]) / 2;
    const d = defaultLen(i);
    return [[j.smpl, Math.min(Math.max(len, LEN_RATIO[0] * d), LEN_RATIO[1] * d)]];
  }));
}

const isLeg = (n) => /_(hip|knee|ankle|foot)$/.test(n);

/**
 * The rest pose (absolute ft, so the figure's scale is 1) for these bone lengths: each default bone
 * keeps its direction (so the T-pose, bone rotations and mesh stay as designed) and takes the new
 * length; the ankles stay at their default height above the floor.
 *
 * Height: the lifted skeleton is ~5% taller than the pitcher's StatsAPI height (its SMPL head
 * joint sits higher than the rig's, whose head mesh reaches 0.63 ft above its head joint). The
 * LEGS keep the delivery's own lengths, because the rig's legs must reach the delivery's feet
 * (scaling them down with the rest left the lead knee jerking: Ohtani stride 2nd difference
 * 6.8 -> 11). Everything above the pelvis (trunk, arms, head) is shrunk by one factor `upper` so
 * the standing height is `heightFt`; the head offset is not scaled.
 * @param lengths from boneLengths (missing joints keep the default length)
 * @param heightFt the pitcher's standing height
 * @returns {{rest: THREE.Vector3[], upper: number}}
 */
export function restFromLengths(lengths, heightFt) {
  const build = (upper) => {
    const rest = [];
    JOINTS.forEach((j, i) => {
      if (j.parent < 0) return rest.push(REST_V[i].clone());
      const len = (lengths[j.smpl] ?? defaultLen(i)) * (isLeg(j.smpl) ? 1 : upper);
      rest.push(rest[j.parent].clone().addScaledVector(REST_V[i].clone().sub(REST_V[j.parent]).normalize(), len));
    });
    const lift = REST_V[J.left_ankle].y - (rest[J.left_ankle].y + rest[J.right_ankle].y) / 2;
    rest.forEach((p) => { p.y += lift; });
    return rest;
  };
  // The pelvis height comes from the legs alone, and the head is `upper` x (head above pelvis at 1)
  // higher: linear, so solve for the head joint height that gives `heightFt`.
  const full = build(1);
  const headY = heightFt - (RIG_HEIGHT_FT - REST_V[J.head].y);
  const upper = (headY - full[J.pelvis].y) / (full[J.head].y - full[J.pelvis].y);
  return { rest: build(upper), upper };
}
