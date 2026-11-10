import * as THREE from "three";
import { toScene } from "./coords.js";
import { stateAt } from "./trajectory.js";

// Baseball: 2.9 in diameter (OBR 3.01: 9-9.25 in circumference).
export const BALL_RADIUS_FT = 1.45 / 12;

/** The solid ball: the Statcast fit, or the simulated path when that mode is chosen. */
export function makeBall() {
  return new THREE.Mesh(
    new THREE.SphereGeometry(BALL_RADIUS_FT, 24, 16),
    new THREE.MeshBasicMaterial({ color: 0xffffff }),
  );
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
