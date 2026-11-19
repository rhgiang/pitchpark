import * as THREE from "three";
import { PALETTE, makeToonMaterial } from "../look.js";
import { groundHeightFt } from "../field.js";
import { toScene } from "../coords.js";

/*
 * Small things on the playing field that are real but tiny. (The rest of the field's details
 * were already painted in ground.js and chalkSegments(): the 5 ft on-deck circles, the coaches'
 * boxes, the 15 ft warning track and the darker lip where grass meets clay. ?old=field hides
 * this file's additions.)
 */

/**
 * The rosin bag: a cloth ball of rosin that pitchers dust their fingers with, kept on the back
 * slope of the mound behind the rubber (OBR 6.02(b) lets the pitcher have it at hand; MLB
 * pitchers' bag sits just behind the rubber, off to the throwing-hand side). Real size is ~3.5 in
 * across; drawn at ~5 in so it reads from the umpire's view.
 * @returns {THREE.Mesh}
 */
export function rosinBag() {
  const [x, y] = [1.7, 62.4]; // Statcast ft: behind the rubber (front edge 60.5, 6 in deep)
  const geo = new THREE.SphereGeometry(0.22, 10, 7);
  geo.scale(1, 0.7, 1);
  const m = new THREE.Mesh(geo, makeToonMaterial(PALETTE.chalk));
  m.position.copy(toScene([x, y, groundHeightFt(x, y) + 0.07]));
  m.castShadow = m.receiveShadow = true;
  m.name = "rosin-bag";
  return m;
}

/** All the field details as one Group. */
export function buildFieldDetails() {
  const g = new THREE.Group();
  g.name = "field-details";
  g.add(rosinBag());
  return g;
}
