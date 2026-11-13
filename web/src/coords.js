import { Vector3 } from "three";

/**
 * Statcast feet (x catcher's right, y plate -> pitcher, z up) to Three.js scene feet
 * (x right, y up, z toward the viewer): (x, y, z) -> (x, z, -y).
 * Both frames are right-handed and this map has determinant +1, so nothing is mirrored:
 * the pitcher sits at negative scene z and the catcher looks down -z, Three's default view axis.
 */
export function toScene([x, y, z]) {
  return new Vector3(x, z, -y);
}
