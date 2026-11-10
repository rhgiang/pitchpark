import * as THREE from "three";
import { toScene } from "./coords.js";
import { buildField, buildZone } from "./field.js";
import { batterEye } from "./cameras.js";
import { stateAt } from "./trajectory.js";
// ponytail: one fixture pitch until the HUD/at-bat picker lands (later tasks use data.js).
import fixture from "../test/fixtures/synthetic.json";

const pitch = fixture.at_bats.flatMap((ab) => ab.pitches).find((p) => p.pitch_type === "CU" && p.real);
const stand = fixture.at_bats.find((ab) => ab.pitches.includes(pitch)).stand;
document.getElementById("label").textContent =
  `Synthetic fixture pitch: ${pitch.pitch_name}, ${pitch.speed_mph.toFixed(1)} mph. Space to throw.`;

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
document.body.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x87b5e0);
scene.add(new THREE.HemisphereLight(0xffffff, 0x445533, 2.5));
scene.add(buildField(), buildZone(pitch.sz_top_ft, pitch.sz_bot_ft));

// Baseball: 2.9 in diameter (OBR 3.01: 9-9.25 in circumference).
const ball = new THREE.Mesh(
  new THREE.SphereGeometry(1.45 / 12, 24, 16),
  new THREE.MeshBasicMaterial({ color: 0xffffff }),
);
scene.add(ball);

// 50 deg vertical FOV fits both the zone and the mound from the box (see cameras.js).
const camera = new THREE.PerspectiveCamera(50, 1, 0.05, 2000);
const eye = batterEye(stand);
camera.position.copy(toScene(eye.position));
camera.lookAt(toScene(eye.target));

function resize() {
  renderer.setSize(window.innerWidth, window.innerHeight);
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
}
window.addEventListener("resize", resize);
resize();

let start = null; // performance.now() at release; null = ball waiting at the release point
window.addEventListener("keydown", (e) => {
  if (e.code === "Space") start = performance.now();
});

renderer.setAnimationLoop((now) => {
  // Real time: 1 s of animation per 1 s of flight, straight from the exported timestamps.
  const t = start === null ? 0 : (now - start) / 1000;
  ball.position.copy(toScene(stateAt(pitch.real, t).p));
  renderer.render(scene, camera);
});
