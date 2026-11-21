import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { J } from "./skeleton.js";

/**
 * Optional sculpted models (scripts/blender/ builds them): public/models/pitcher.glb (body and cap,
 * rest T-pose, skinned to the skeleton.js bones by name; make_pitcher.py), mitt.glb (glove hand)
 * and grip.glb (throwing hand round the ball). The hands: Same local frame as figure.js's procedural
 * hands: built for the LEFT side, scene ft, origin at the wrist, +x along the hand, +y the back of
 * the hand, +z the thumb side; the ball's centre at (0.28, 0, 0). Each mesh's material name picks
 * the figure colour (glove, gloveDark, gloveLight, skin, ...). A missing file, a load error, or
 * ?models=0 keeps the procedural shapes.
 */
const cache = {};
const NAMES = ["pitcher", "mitt", "grip"];

/** Load the models once (resolves even on failure). Call before makePitcher. */
export function loadHandModels(base = "/models/") {
  if (typeof location !== "undefined" && new URLSearchParams(location.search).get("models") === "0") return Promise.resolve();
  const loader = new GLTFLoader();
  return Promise.all(NAMES.map((n) => loader.loadAsync(`${base}${n}.glb`)
    .then((gltf) => { cache[n] = toParts(gltf.scene); })
    .catch(() => {}))); // ponytail: silent fallback; the procedural hand is always valid
}

/** [material key, fresh geometry][] for `name`, or null if not loaded. */
export const handModel = (name) => cache[name]?.map(([k, g]) => [k, g.clone()]) ?? null;

/**
 * Flatten a glTF scene to [material key, geometry] with only the attributes the figure's merge
 * expects (indexed position, normal, uv), transforms baked in. A skinned mesh also keeps its skin
 * weights, its joint indices turned into skeleton.js indices by bone name (the glTF's joint order
 * is the exporter's; the figure has its own bones at the same rest pose).
 */
export function toParts(scene) {
  scene.updateMatrixWorld(true);
  const parts = [];
  scene.traverse((o) => {
    if (!o.isMesh) return;
    const src = o.geometry.clone().applyMatrix4(o.matrixWorld);
    const g = new THREE.BufferGeometry();
    for (const a of ["position", "normal", "uv"]) if (src.attributes[a]) g.setAttribute(a, src.attributes[a]);
    if (!g.attributes.normal) g.computeVertexNormals();
    if (!g.attributes.uv) g.setAttribute("uv", new THREE.Float32BufferAttribute(new Float32Array(g.attributes.position.count * 2), 2));
    g.setIndex(src.index ?? [...Array(g.attributes.position.count).keys()]);
    if (o.isSkinnedMesh) {
      const map = o.skeleton.bones.map((b) => J[b.name]);
      if (map.some((j) => j === undefined)) throw new Error(`unknown bone in ${o.name}`);
      const si = src.attributes.skinIndex, sw = src.attributes.skinWeight; // weights may be normalised ints
      const get = ["getX", "getY", "getZ", "getW"];
      g.setAttribute("skinIndex", new THREE.Uint16BufferAttribute(Uint16Array.from({ length: si.count * 4 }, (_, k) => map[si[get[k % 4]](k >> 2)]), 4));
      g.setAttribute("skinWeight", new THREE.Float32BufferAttribute(Float32Array.from({ length: sw.count * 4 }, (_, k) => sw[get[k % 4]](k >> 2)), 4));
    }
    parts.push([o.material?.name || "skin", g]);
  });
  return parts.length ? parts : null;
}
