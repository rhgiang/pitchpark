// Movement aids: the fading trail behind the ball and the dashed "no spin" reference line.
// Both are thin tubes along an exported path (no physics here), built once per pitch and then
// revealed up to the playback time by one uniform, so they grow with the ball and never run ahead
// of it (the spoiler rule).
import * as THREE from "three";
import { toScene } from "./coords.js";
import { inkProof } from "./look.js";
import { BALL_RADIUS_FT } from "./ball.js";

const RING = 6; // vertices around the tube

const vertexShader = /* glsl */ `
  attribute float aT;
  varying float vT;
  void main() {
    vT = aT;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }`;

// Drawn up to uT (s since release). Fade: older parts dim to 25% with time constant uFade (s),
// so the whole curve stays readable after the pitch lands. Dash: on/off every uDash s of flight.
const fragmentShader = /* glsl */ `
  uniform vec3 uColor;
  uniform float uT, uFade, uDash, uOpacity;
  varying float vT;
  void main() {
    if (vT > uT) discard;
    if (uDash > 0.0 && fract(vT / uDash) > 0.55) discard;
    float a = uOpacity;
    if (uFade > 0.0) a *= 0.25 + 0.75 * exp(-(uT - vT) / uFade);
    gl_FragColor = vec4(uColor, a);
  }`;

/**
 * A tube that draws an exported path up to the playback time.
 * @param color hex; radiusFt tube radius; fadeS fade time constant (0: no fade); dashS dash
 *   period in s of flight (0: solid). Ink-proof (layer 2) like the ball: no outline crosses it.
 */
export function makeTube({ color, radiusFt, fadeS = 0, dashS = 0, opacity = 1 }) {
  const mesh = new THREE.Mesh(new THREE.BufferGeometry(), new THREE.ShaderMaterial({
    vertexShader, fragmentShader, transparent: true, depthWrite: false,
    uniforms: {
      uColor: { value: new THREE.Color(color) }, uT: { value: 0 }, uFade: { value: fadeS },
      uDash: { value: dashS }, uOpacity: { value: opacity },
    },
  }));
  mesh.frustumCulled = false; // the geometry changes per pitch; the bounds would go stale
  mesh.userData = { radiusFt, path: null, t: [], tEnd: 0 };
  mesh.visible = false;
  return inkProof(mesh);
}

/**
 * Show `path` ({t, p} Statcast ft) on the tube, or hide it (null), then reveal up to t_s.
 * The geometry is rebuilt only when the path changes.
 */
export function updateTube(tube, path, t_s) {
  const u = tube.userData;
  if (path !== u.path) build(tube, path);
  tube.visible = path != null;
  if (!path) return;
  const t = Math.min(t_s, u.tEnd);
  tube.material.uniforms.uT.value = t;
  // Draw only the segments that start before t (the shader trims the last one exactly), so the
  // hidden part doesn't mask outlines in look.js's ink-proof pass either.
  let k = 0;
  while (k < u.t.length - 1 && u.t[k] < t) k++;
  tube.geometry.setDrawRange(0, k * RING * 6);
}

function build(tube, path) {
  const u = tube.userData;
  u.path = path;
  tube.geometry.dispose();
  tube.geometry = new THREE.BufferGeometry();
  if (!path) return;
  const n = path.t.length;
  u.t = path.t;
  // Balls in the dirt: the trail stops where the ball is hidden (centre below one radius, ball.js).
  const dirt = path.p.findIndex((q) => q[2] < BALL_RADIUS_FT);
  u.tEnd = dirt < 0 ? path.t[n - 1] : path.t[dirt];
  const pts = path.p.map(toScene);
  const pos = new Float32Array(n * RING * 3);
  const at = new Float32Array(n * RING);
  const up = new THREE.Vector3(0, 1, 0);
  const tan = new THREE.Vector3();
  const nrm = new THREE.Vector3();
  const bin = new THREE.Vector3();
  for (let k = 0; k < n; k++) {
    tan.subVectors(pts[Math.min(n - 1, k + 1)], pts[Math.max(0, k - 1)]).normalize();
    nrm.crossVectors(tan, up).normalize(); // the flight is never vertical, so this is safe
    bin.crossVectors(nrm, tan);
    for (let j = 0; j < RING; j++) {
      const a = (2 * Math.PI * j) / RING;
      const v = pts[k].clone()
        .addScaledVector(nrm, u.radiusFt * Math.cos(a))
        .addScaledVector(bin, u.radiusFt * Math.sin(a));
      pos.set([v.x, v.y, v.z], (k * RING + j) * 3);
      at[k * RING + j] = path.t[k];
    }
  }
  const index = [];
  for (let k = 0; k < n - 1; k++) {
    for (let j = 0; j < RING; j++) {
      const a = k * RING + j;
      const b = k * RING + ((j + 1) % RING);
      index.push(a, a + RING, b, b, a + RING, b + RING);
    }
  }
  tube.geometry.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  tube.geometry.setAttribute("aT", new THREE.BufferAttribute(at, 1));
  tube.geometry.setIndex(index);
}
