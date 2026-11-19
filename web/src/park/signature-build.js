import * as THREE from "three";
import { isOld } from "./old.js";
import { buildTower } from "./tower.js";
import { buildFalls } from "./falls.js";
import { buildOracle } from "./oracle.js";

/**
 * All signature features as one Group, plus update hooks. flags/poles are the park's shared flag lists (the
 * pennants join the one animated flag mesh). group.userData.setSpin(spin) turns the tower's ball.
 */
export function buildSignature(flags, poles) {
  const g = new THREE.Group();
  g.name = "signature";
  g.userData.setSpin = () => {};
  if (!isOld("signature")) {
    const t = buildTower(flags, poles);
    g.add(t);
    g.userData.setSpin = t.userData.setSpin;
  }
  if (!isOld("signature")) {
    g.add(buildFalls(), buildOracle(flags, poles)); // each draws its first-built look under ?old=falls / ?old=oracle
  }
  return g;
}
