import { test } from "node:test";
import assert from "node:assert/strict";
import { Box3, Vector3 } from "three";
import { JOINTS, J, boneLengths, restFromLengths } from "../src/skeleton.js";
import { makePitcher, poseAtRelease } from "../src/pitcher.js";
import { figureGeometry } from "../src/figure.js";

// SYNTHETIC delivery (a test fixture, not a real pitcher): the default rig's bones, each stretched
// by a known ratio (left and right differ, as a noisy lift's would), repeated for 40 frames with
// small noise, and one depth-collapsed frame (every bone 40% of its length).
const RATIO = { hip: [1.2, 1.1], knee: [1.15, 1.1], ankle: [1.1, 1.1], elbow: [1.0, 1.0], wrist: [1.2, 1.2], spine2: [1.3, 1.3] };
const ratioOf = (name) => {
  const side = name.startsWith("left_") ? 0 : 1;
  const key = name.replace(/^(left|right)_/, "");
  return RATIO[key]?.[side] ?? 1;
};
const REST = JOINTS.map((j) => new Vector3(...j.rest));
function frame(scale = 1, noise = 0, seed = 1) {
  let s = seed; const rnd = () => ((s = (s * 16807) % 2147483647) / 2147483647 - 0.5) * 2;
  const p = [];
  JOINTS.forEach((j, i) => {
    if (j.parent < 0) return p.push(REST[0].clone());
    const off = REST[i].clone().sub(REST[j.parent]).multiplyScalar(ratioOf(j.smpl) * scale);
    p.push(p[j.parent].clone().add(off));
  });
  return p.map((v) => [v.x + noise * rnd(), v.z + noise * rnd(), v.y + noise * rnd()]); // Statcast ft: [x, y(depth), z(up)]
}
const DELIVERY = { joints_ft: [...Array.from({ length: 40 }, (_, i) => frame(1, 0.01, i + 1)), frame(0.4)] };
const dist = (a, b) => REST[a].distanceTo(REST[b]);

test("boneLengths: median over frames (a collapsed frame is ignored), left/right averaged, within tolerance", () => {
  const L = boneLengths(DELIVERY);
  for (const [name, expect] of [["left_knee", (1.15 + 1.1) / 2 * dist(J.left_knee, J.left_hip)], ["right_wrist", 1.2 * dist(J.right_wrist, J.right_elbow)]]) {
    assert.ok(Math.abs(L[name] - expect) < 0.03, `${name}: ${L[name]} vs ${expect}`);
  }
  assert.ok(Math.abs(L.left_knee - L.right_knee) < 1e-9, "left and right bones come back equal");
  assert.ok(Math.abs(L.left_hip - L.right_hip) < 1e-9);
  // spine2 at 1.3x would be 0.585 ft; the clamp keeps every bone within 0.8-1.25x of the rig's.
  assert.ok(L.spine2 <= 1.25 * dist(J.spine2, J.spine1) + 1e-9);
});

test("a figure built from the lengths has those bones and his standing height", () => {
  const L = boneLengths(DELIVERY);
  const heightIn = 78;
  const fig = makePitcher("R", heightIn, L);
  poseAtRelease(fig, [-2.4, 54.3, 5.7], 40);
  fig.updateMatrixWorld(true);
  const at = (n) => fig.getObjectByName(JOINTS[J[n]].mixamo).getWorldPosition(new Vector3());
  // Legs keep the delivery's lengths; the trunk and arms share one factor to reach his height.
  assert.ok(Math.abs(at("left_hip").distanceTo(at("left_knee")) - L.left_knee) < 1e-3);
  assert.ok(Math.abs(at("right_knee").distanceTo(at("right_ankle")) - L.right_ankle) < 1e-3);
  // The rest-pose mesh's top (cap) is his height.
  const top = new Box3().setFromBufferAttribute(figureGeometry("R", restFromLengths(L, heightIn / 12).rest).geometry.attributes.position).max.y;
  assert.ok(Math.abs(top - heightIn / 12) < 0.05, `top ${top} ft vs ${heightIn / 12}`);
  // Longer legs than the default rig of the same height: his proportions, not a scaled copy.
  const stock = makePitcher("R", heightIn);
  poseAtRelease(stock, [-2.4, 54.3, 5.7], 40);
  stock.updateMatrixWorld(true);
  const leg = (f) => f.getObjectByName(JOINTS[J.left_hip].mixamo).getWorldPosition(new Vector3())
    .distanceTo(f.getObjectByName(JOINTS[J.left_knee].mixamo).getWorldPosition(new Vector3()));
  assert.ok(leg(fig) > leg(stock) * 1.05);
});

test("without lengths the rig is the default one, scaled by height / 75", () => {
  const fig = makePitcher("R", 75);
  assert.equal(fig.userData.scale, 1);
  assert.equal(makePitcher("L", 78).userData.scale, 78 / 75);
  const top = new Box3().setFromBufferAttribute(figureGeometry("R").geometry.attributes.position).max.y;
  assert.ok(Math.abs(top - 75 / 12) < 0.05);
});
