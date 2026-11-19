import * as THREE from "three";
import { toScene } from "./coords.js";
import { stateAt } from "./trajectory.js";
import { makeToonMaterial } from "./look.js";

// Baseball: 2.9 in diameter (OBR 3.01: 9-9.25 in circumference).
export const BALL_RADIUS_FT = 1.45 / 12;

/*
 * Seams. A baseball's cover is two figure-8 pieces of leather sewn together; the seam between them
 * is one closed curve, well approximated by the "tennis-ball curve" on the unit sphere:
 *   p(t) = (cos t + k cos 3t, sin t - k sin 3t, 2 sqrt(k (1 - k)) sin 2t), normalised.
 * It is sewn with 108 double stitches (OBR 3.01), red on a real MLB ball.
 *
 * Seam orientation: how the seams sit relative to the spin axis. It sets how many seams cross the
 * ball's spinning equator per turn: 4 (a "four-seam" fastball) or 2 ("two-seam"), and through the
 * seam-shifted wake (Barton Smith; Driveline) it bends the pitch beyond Magnus. Statcast does not
 * publish it, so the viewer ASSUMES it from the pitch type (seamStyle) and says so on screen.
 * M3 can later choose it per pitch from the measured vs movement-implied spin axis gap.
 */
const SEAM_K = 0.3; // lobe shape (0.3: lobes never give more than 4 seams per turn, as on a real ball)
const STITCHES = 108;
/** Spin axes in the seam curve's own frame: four-seam (z) and two-seam (the x = y diagonal). */
export const SEAM_AXES = { 4: [0, 0, 1], 2: [Math.SQRT1_2, Math.SQRT1_2, 0] };
const TWO_SEAM_TYPES = new Set(["SI", "FT"]); // sinker / two-seamer; everything else four-seam

/** Assumed seam orientation for a Statcast pitch_type: 2 (two-seam) or 4 (four-seam). */
export const seamStyle = (pitchType) => (TWO_SEAM_TYPES.has(pitchType) ? 2 : 4);

/** n points evenly spaced (by arc length) round the seam, as unit vectors [x, y, z]. */
export function seamCurve(n) {
  const fine = [];
  for (let i = 0; i < 2000; i++) {
    const t = (2 * Math.PI * i) / 2000;
    const p = [Math.cos(t) + SEAM_K * Math.cos(3 * t), Math.sin(t) - SEAM_K * Math.sin(3 * t),
      2 * Math.sqrt(SEAM_K * (1 - SEAM_K)) * Math.sin(2 * t)];
    const r = Math.hypot(...p);
    fine.push(p.map((v) => v / r));
  }
  const seg = fine.map((p, i) => Math.hypot(...p.map((v, j) => v - fine[(i + 1) % fine.length][j])));
  const total = seg.reduce((a, b) => a + b, 0);
  const out = [];
  let i = 0, run = 0;
  for (let k = 0; k < n; k++) { // walk the fine polyline, dropping a point every total / n
    const want = (k * total) / n;
    while (run + seg[i] < want) run += seg[i++];
    const f = (want - run) / seg[i], a = fine[i], b = fine[(i + 1) % fine.length];
    const p = a.map((v, j) => v + (b[j] - v) * f);
    const r = Math.hypot(...p);
    out.push(p.map((v) => v / r));
  }
  return out;
}

/** How many times the seam crosses the great circle perpendicular to `axis` (seams per turn). */
export function seamCrossings(axis, n = 4000) {
  const pts = seamCurve(n);
  const d = pts.map((p) => p[0] * axis[0] + p[1] * axis[1] + p[2] * axis[2]);
  return d.filter((v, i) => (v >= 0) !== (d[(i + 1) % n] >= 0)).length; // sign changes
}

/**
 * The stitches: at each of 108 stations round the seam, a V of two short red stitches, one from
 * each cover, meeting at the seam and pointing along it. One instanced box, 216 copies.
 */
function makeStitches(r) {
  const pts = seamCurve(STITCHES).map((p) => new THREE.Vector3(...p));
  const box = new THREE.BoxGeometry(1, 1, 1);
  const mesh = new THREE.InstancedMesh(box, makeToonMaterial(0xc8202a, { surface: "none" }), 2 * STITCHES);
  const m = new THREE.Matrix4(), x = new THREE.Vector3(), z = new THREE.Vector3();
  const W = 0.13, H = 0.1, LIFT = 1.008; // stitch spread across / along the seam (x r), height off the leather
  let k = 0;
  pts.forEach((p, i) => {
    const t = pts[(i + 1) % STITCHES].clone().sub(pts[(i + STITCHES - 1) % STITCHES]).normalize(); // along
    const s = p.clone().cross(t).normalize(); // across the seam, on the surface
    for (const side of [1, -1]) {
      const a = p.clone().addScaledVector(s, side * W).addScaledVector(t, -H).normalize().multiplyScalar(r * LIFT);
      const b = p.clone().addScaledVector(s, side * W * 0.15).normalize().multiplyScalar(r * LIFT);
      x.subVectors(b, a);
      const len = x.length();
      x.normalize();
      const y = a.clone().add(b).normalize(); // outward
      z.crossVectors(x, y);
      m.makeBasis(x, y, z).scale(new THREE.Vector3(len, r * 0.035, r * 0.035)).setPosition(a.clone().add(b).multiplyScalar(0.5));
      mesh.setMatrixAt(k++, m);
    }
  });
  return mesh;
}

/**
 * The solid ball: the Statcast fit, or the simulated path when that mode is chosen. Off-white
 * leather with the red stitched seam; the sphere's poles (local y) are the spin axis, and a blue
 * rod through them shows it. children: [rod, seams] (setSeams turns the seams).
 */
export function makeBall() {
  const r = BALL_RADIUS_FT;
  const ball = new THREE.Mesh(new THREE.SphereGeometry(r, 32, 20), makeToonMaterial(0xf2efe6, { surface: "none" }));
  const rod = new THREE.Mesh(
    new THREE.CylinderGeometry(0.012, 0.012, 4 * r, 8),
    new THREE.MeshBasicMaterial({ color: 0x1144ff }),
  );
  const seams = new THREE.Group();
  const line = seamCurve(240).map((p) => new THREE.Vector3(...p).multiplyScalar(r * 1.002));
  seams.add(new THREE.LineLoop(new THREE.BufferGeometry().setFromPoints(line), new THREE.LineBasicMaterial({ color: 0xc9bfae })));
  seams.add(makeStitches(r));
  ball.add(rod, seams);
  setSeams(ball, 4);
  return ball;
}

/** Turn the seams so the chosen orientation's axis (SEAM_AXES) lies on the spin axis (local y). */
export function setSeams(ball, style) {
  if (ball.userData.seams === style) return;
  ball.children[1].quaternion.setFromUnitVectors(new THREE.Vector3(...SEAM_AXES[style]), Y);
  ball.userData.seams = style;
}

const Y = new THREE.Vector3(0, 1, 0);

/**
 * Orient the ball's pole along the spin axis and turn it about that axis by angle_rad.
 * Display-only port of pitchsim/physics.py spin_unit_vector (Statcast spin_axis: 180 deg = pure
 * backspin, 0 = topspin; w_hat = (cos th, 0, sin th) in Statcast x, y, z, i.e. (cos th, sin th, 0)
 * in scene axes). No physics: the exported path is what flies. Null axis: no rotation, rod hidden.
 */
export function spinBall(ball, spin_axis_deg, angle_rad) {
  ball.children[0].visible = spin_axis_deg != null;
  if (spin_axis_deg == null) return;
  const th = (spin_axis_deg * Math.PI) / 180;
  const w = new THREE.Vector3(Math.cos(th), Math.sin(th), 0);
  ball.quaternion.setFromUnitVectors(Y, w)
    .multiply(new THREE.Quaternion().setFromAxisAngle(Y, angle_rad));
}

/** Ghost: the simulated (M1 + M3) ball, translucent, flown beside the Statcast fit. */
export function makeGhost() {
  return new THREE.Mesh(
    new THREE.SphereGeometry(BALL_RADIUS_FT, 24, 16),
    // depthWrite off so the solid ball behind it still draws.
    new THREE.MeshBasicMaterial({ color: 0xffb070, transparent: true, opacity: 0.45, depthWrite: false }),
  );
}

/** True if the path ends with the ball's centre below one radius: it hit the dirt. */
export function inDirt(path) {
  return path.p[path.p.length - 1][2] < BALL_RADIUS_FT;
}

/**
 * Place `ball` on `path` at t_s; hide it when there is no path.
 * Balls in the dirt: Statcast's constant-acceleration fit carries on below the ground (plate z
 * down to about -0.8 ft) where the real ball bounced. We do not draw that part: the ball is
 * hidden once its centre is below one radius. Display only; the exported data is unchanged.
 */
export function updateBall(ball, path, t_s) {
  if (!path) {
    ball.visible = false;
    return;
  }
  const { p } = stateAt(path, t_s);
  ball.visible = p[2] >= BALL_RADIUS_FT;
  ball.position.copy(toScene(p));
}
