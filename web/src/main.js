import * as THREE from "three";
import { toScene } from "./coords.js";
import { buildField, buildZone } from "./field.js";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import {
  batterEye, sideView, rideAlong, BATTER_EYE_VFOV_DEG, SIDE_VFOV_DEG, RIDE_VFOV_DEG, FREE_VFOV_DEG,
  FREE_START, FREE_MIN_DIST_FT, FREE_MAX_DIST_FT,
} from "./cameras.js";
import { loadIndex, loadPitcher } from "./data.js";
import {
  initialState, reduce, currentAtBat, currentPitch, shownPaths, MODES, SPEEDS, VIEWS, spinFactor,
} from "./state.js";
import { makeBall, makeGhost, updateBall, spinBall } from "./ball.js";
import { renderHud } from "./hud.js";

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
document.body.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x87b5e0);
scene.add(new THREE.HemisphereLight(0xffffff, 0x445533, 2.5));
scene.add(buildField());
const ball = makeBall();
const ghost = makeGhost();
scene.add(ball, ghost);
let zone = null;

// Vertical FOV chosen so every pitch's zone, plate arrival and release fit (see cameras.js).
const camera = new THREE.PerspectiveCamera(BATTER_EYE_VFOV_DEG, 1, 0.05, 2000);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enabled = false;
controls.minDistance = FREE_MIN_DIST_FT;
controls.maxDistance = FREE_MAX_DIST_FT;
let freePose = { position: toScene(FREE_START.position), target: toScene(FREE_START.target) };
let appliedView = null;
const VFOV = { batter: BATTER_EYE_VFOV_DEG, side: SIDE_VFOV_DEG, free: FREE_VFOV_DEG, ride: RIDE_VFOV_DEG };

/** Point the camera for the current view. Runs every frame: cheap, and ride follows t. */
function updateCamera() {
  const ab = currentAtBat(state);
  const path = shownPaths(state)[0]; // none for no_tracking: keep the last pose
  if (state.view !== appliedView) {
    // Leaving Free: remember the orbit (views 1/2/4 overwrite the camera); entering: restore it.
    if (appliedView === "free") freePose = { position: camera.position.clone(), target: controls.target.clone() };
    appliedView = state.view;
    camera.fov = VFOV[state.view];
    camera.updateProjectionMatrix();
    controls.enabled = state.view === "free";
    if (state.view === "free") { // first visit: FREE_START; later: where the user left it
      camera.position.copy(freePose.position);
      controls.target.copy(freePose.target);
    }
  }
  if (state.view === "free") { controls.update(); return; }
  let pose = null;
  if (state.view === "batter") pose = batterEye(ab.stand);
  else if (path) pose = state.view === "side" ? sideView(path, camera.aspect) : rideAlong(path, state.t);
  if (!pose) return;
  camera.position.copy(toScene(pose.position));
  camera.lookAt(toScene(pose.target));
}

function resize() {
  renderer.setSize(window.innerWidth, window.innerHeight);
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
}
window.addEventListener("resize", resize);
resize();

const $ = (id) => document.getElementById(id);
let state = initialState();
let shown = { ab: null, pitch: null }; // what the camera and zone are set up for

/** Apply an action; refresh the scene set-up and controls when the pitch or at-bat changed. */
function dispatch(action) {
  const before = state;
  state = reduce(state, action);
  if (action.type === "tick") {
    // The pitch just landed: re-render so the result (and "in the dirt") appears now.
    if (before.playing && !state.playing) renderHud(state, currentPitch(state));
    return;
  }
  const ab = currentAtBat(state);
  const pitch = currentPitch(state);
  if (pitch !== shown.pitch) { // zone height is per pitch (Statcast sz_top / sz_bot)
    if (zone) scene.remove(zone);
    zone = pitch.sz_top_ft == null ? null : buildZone(pitch.sz_top_ft, pitch.sz_bot_ft);
    if (zone) scene.add(zone);
  }
  shown = { ab, pitch };
  $("atbat").value = String(state.abIndex);
  for (const m of MODES) {
    const b = $(`mode-${m}`);
    b.disabled = m !== "real" && !state.modeAllowed;
    b.classList.toggle("on", state.mode === m);
  }
  for (const v of VIEWS) $(`view-${v}`).classList.toggle("on", state.view === v);
  for (const v of SPEEDS) $(`speed-${v}`).classList.toggle("on", state.speed === v);
  renderHud(state, pitch);
}

const randomSeed = () => Math.floor(Math.random() * 2 ** 31);

async function pickPitcher(slug) {
  dispatch({ type: "pickPitcher", slug, data: await loadPitcher(slug) });
  $("pitcher").value = slug;
  // ponytail: plain <option> list; a search box if at-bat lists grow past a few hundred.
  $("atbat").replaceChildren(...state.data.at_bats.map((ab, i) => new Option(
    // No outcome in the label: it would spoil the at-bat.
    `${ab.game_date} · ${ab.top ? "Top" : "Bot"} ${ab.inning} · ${ab.stand}HB`,
    String(i))));
  dispatch({ type: "pickAtBat", index: 0 });
}

// Every action has a key and a button (headless screenshots cannot press keys).
const KEYS = {
  Space: { type: "replay" }, ArrowRight: { type: "nextPitch" }, ArrowLeft: { type: "prevPitch" },
  KeyR: { type: "setMode", mode: "real" }, KeyS: { type: "setMode", mode: "sim" },
  KeyG: { type: "setMode", mode: "ghost" },
  Digit1: { type: "setView", view: "batter" }, Digit2: { type: "setView", view: "side" },
  Digit3: { type: "setView", view: "free" }, Digit4: { type: "setView", view: "ride" },
  KeyZ: { type: "setSpeed", speed: 1 }, KeyX: { type: "setSpeed", speed: 0.25 },
  KeyC: { type: "setSpeed", speed: 0.1 },
};
window.addEventListener("keydown", (e) => {
  // Leave browser shortcuts (Ctrl+R, ...) alone, and keys aimed at a focused control: a select
  // uses arrows itself, and Space on a focused button would also click it (a double replay).
  if (!state.data || e.ctrlKey || e.metaKey || e.altKey) return;
  if (["SELECT", "BUTTON"].includes(e.target.tagName)) return;
  if (e.code === "KeyA") return dispatch({ type: "pickAtBat", index: "random", seed: randomSeed() });
  if (KEYS[e.code]) {
    e.preventDefault();
    dispatch(KEYS[e.code]);
  }
});
// Drop focus after a click or a pick so the keys work again straight away.
document.addEventListener("click", (e) => { if (e.target.tagName === "BUTTON") e.target.blur(); });
document.addEventListener("change", (e) => e.target.blur());
$("prev").onclick = () => dispatch({ type: "prevPitch" });
$("replay").onclick = () => dispatch({ type: "replay" });
$("next").onclick = () => dispatch({ type: "nextPitch" });
$("random").onclick = () => dispatch({ type: "pickAtBat", index: "random", seed: randomSeed() });
for (const m of MODES) $(`mode-${m}`).onclick = () => dispatch({ type: "setMode", mode: m });
for (const v of VIEWS) $(`view-${v}`).onclick = () => dispatch({ type: "setView", view: v });
for (const v of SPEEDS) $(`speed-${v}`).onclick = () => dispatch({ type: "setSpeed", speed: v });
$("pitcher").onchange = (e) => pickPitcher(e.target.value);
$("atbat").onchange = (e) => dispatch({ type: "pickAtBat", index: Number(e.target.value) });

// Initial state from the URL, e.g. ?pitcher=sale-chris&ab=0&pitch=2&t=0.3&mode=ghost (frozen at
// t, for screenshots); ab=random&seed=N picks a seeded random at-bat.
async function start() {
  const q = new URLSearchParams(location.search);
  const index = await loadIndex();
  $("pitcher").replaceChildren(...index.map((p) => new Option(`${p.name} (${p.throws}HP)`, p.slug)));
  await pickPitcher(q.get("pitcher") ?? index[0].slug);
  if (q.has("ab")) {
    const ab = q.get("ab");
    dispatch(ab === "random"
      ? { type: "pickAtBat", index: "random", seed: Number(q.get("seed") ?? 0) }
      : { type: "pickAtBat", index: Number(ab) });
  }
  for (let i = 0; i < Number(q.get("pitch") ?? 0); i++) dispatch({ type: "nextPitch" });
  // Unknown mode/speed values are ignored by the reducer.
  if (q.has("mode")) dispatch({ type: "setMode", mode: q.get("mode") });
  if (q.has("view")) dispatch({ type: "setView", view: q.get("view") });
  if (q.has("speed")) dispatch({ type: "setSpeed", speed: Number(q.get("speed")) });
  // Freeze at t (no playback) so a headless screenshot catches the ball mid-flight.
  state = { ...state, playing: false, t: Number(q.get("t") ?? 0) };
  renderHud(state, currentPitch(state)); // t may already be past landing
}
start();

let last = null;
renderer.setAnimationLoop((now) => {
  // Real time: 1 s of animation per 1 s of flight (times speed), from the exported timestamps.
  // dt is capped so a backgrounded tab does not jump the ball to the plate on return.
  const dt = last === null ? 0 : Math.min((now - last) / 1000, 0.1);
  last = now;
  if (state.data) {
    dispatch({ type: "tick", dt });
    const p = currentPitch(state);
    updateBall(ball, state.mode === "sim" ? p.sim : p.real, state.t);
    spinBall(ball, p.spin_rpm == null ? null : p.spin_axis_deg, 2 * Math.PI * (p.spin_rpm / 60) * state.t * spinFactor(state.speed));
    updateBall(ghost, state.mode === "ghost" ? p.sim : null, state.t);
  }
  if (state.data) updateCamera();
  renderer.render(scene, camera);
});
