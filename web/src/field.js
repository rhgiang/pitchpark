import * as THREE from "three";
import { toScene } from "./coords.js";

// Geometry from the Official Baseball Rules (2.01-2.02, Appendix 2). Statcast's y = 0 is the
// back point of home plate, which is also where the rules measure the rubber from.
const IN = 1 / 12;
const PLATE_FRONT_Y_FT = 17 * IN;
export const RUBBER_FRONT_Y_FT = 60.5; // front edge of the rubber
const RUBBER_W_FT = 24 * IN;
const RUBBER_D_FT = 6 * IN;
const MOUND_HEIGHT_FT = 10 * IN; // rubber is 10 in above home plate
const MOUND_CENTRE_Y_FT = 59; // 18 ft circle centred 59 ft from the back of the plate
const MOUND_RADIUS_FT = 9;
const MOUND_TOP_RADIUS_FT = 2.5;

/**
 * Ground height (ft above home plate) at Statcast (x, y), matching the mound cone built below:
 * flat on top out to 2.5 ft from its centre, then falling linearly to 0 at 9 ft.
 */
export function groundHeightFt(x, y) {
  const r = Math.hypot(x, y - MOUND_CENTRE_Y_FT);
  const f = (MOUND_RADIUS_FT - r) / (MOUND_RADIUS_FT - MOUND_TOP_RADIUS_FT);
  return MOUND_HEIGHT_FT * Math.min(1, Math.max(0, f));
}

/** Ground, home plate, mound and rubber, plain materials. */
export function buildField() {
  const g = new THREE.Group();

  const ground = new THREE.Mesh(
    new THREE.PlaneGeometry(400, 400),
    new THREE.MeshLambertMaterial({ color: 0x4f7d3a }),
  );
  ground.rotation.x = -Math.PI / 2;
  g.add(ground);

  // Home plate: a pentagon 17 in across the front, 8.5 in straight sides, then two 12 in edges
  // meeting at the back point (0, 0).
  const w = 8.5 * IN;
  const outline = [[-w, PLATE_FRONT_Y_FT], [w, PLATE_FRONT_Y_FT], [w, w], [0, 0], [-w, w]];
  const shape = new THREE.Shape(outline.map(([x, y]) => new THREE.Vector2(x, y)));
  const plate = new THREE.Mesh(
    new THREE.ShapeGeometry(shape),
    new THREE.MeshLambertMaterial({ color: 0xffffff }),
  );
  // Shape is drawn in Statcast (x, y); lay it flat: shape y -> Statcast y -> scene -z.
  plate.rotation.x = -Math.PI / 2;
  plate.position.y = 0.01; // just above the ground, no z-fighting
  g.add(plate);

  // Mound: a low flattened cone; real mounds slope 1 in per ft from 6 in in front of the rubber.
  const mound = new THREE.Mesh(
    new THREE.CylinderGeometry(MOUND_TOP_RADIUS_FT, MOUND_RADIUS_FT, MOUND_HEIGHT_FT, 48),
    new THREE.MeshLambertMaterial({ color: 0xa0704a }),
  );
  mound.position.copy(toScene([0, MOUND_CENTRE_Y_FT, MOUND_HEIGHT_FT / 2]));
  g.add(mound);

  const rubber = new THREE.Mesh(
    new THREE.BoxGeometry(RUBBER_W_FT, 0.05, RUBBER_D_FT),
    new THREE.MeshLambertMaterial({ color: 0xffffff }),
  );
  rubber.position.copy(toScene([0, RUBBER_FRONT_Y_FT + RUBBER_D_FT / 2, MOUND_HEIGHT_FT + 0.025]));
  g.add(rubber);

  return g;
}

/** Strike zone outline at the front of the plate: 17 in wide, from szBot to szTop ft. */
export function buildZone(szTop, szBot) {
  const w = 8.5 * IN;
  const pts = [[-w, szBot], [w, szBot], [w, szTop], [-w, szTop]].map(([x, z]) =>
    toScene([x, PLATE_FRONT_Y_FT, z]),
  );
  return new THREE.LineLoop(
    new THREE.BufferGeometry().setFromPoints(pts),
    new THREE.LineBasicMaterial({ color: 0xffffff }),
  );
}
