import * as THREE from "three";
import { toScene } from "./coords.js";
import { stateAt } from "./trajectory.js";

// Baseball: 2.9 in diameter (OBR 3.01: 9-9.25 in circumference).
export const BALL_RADIUS_FT = 1.45 / 12;

/** The solid ball: the Statcast fit, or the simulated path when that mode is chosen. */
export function makeBall() {
  // White with red meridian stripes and an equator band, so rotation is visible. The sphere's
  // poles (local y) are the spin axis; a blue rod through them shows it.
  const c = document.createElement("canvas");
  c.width = 128; c.height = 64;
  const g = c.getContext("2d");
  g.fillStyle = "#fff"; g.fillRect(0, 0, 128, 64);
  g.fillStyle = "#c33";
  for (const x of [0, 32, 64, 96]) g.fillRect(x, 0, 4, 64);
  g.fillRect(0, 30, 128, 4);
  const ball = new THREE.Mesh(
    new THREE.SphereGeometry(BALL_RADIUS_FT, 24, 16),
    new THREE.MeshBasicMaterial({ map: new THREE.CanvasTexture(c) }),
  );
  const rod = new THREE.Mesh(
    new THREE.CylinderGeometry(0.012, 0.012, 4 * BALL_RADIUS_FT, 8),
    new THREE.MeshBasicMaterial({ color: 0x1144ff }),
  );
  ball.add(rod);
  return ball;
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
