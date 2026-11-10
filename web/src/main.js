import "./ui.css";
import * as THREE from "three";
import { toScene } from "./coords.js";
import { buildField, buildZone } from "./field.js";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import {
  batterEye, catcherView, sideView, rideAlong, BATTER_EYE_VFOV_DEG, CATCHER_VFOV_DEG, SIDE_VFOV_DEG,
  RIDE_VFOV_DEG, FREE_VFOV_DEG, FREE_START, FREE_MIN_DIST_FT, FREE_MAX_DIST_FT,
} from "./cameras.js";
import { loadIndex, loadPitcher } from "./data.js";
import {
  initialState, reduce, currentAtBat, currentPitch, shownPaths, MODES, SPEEDS, VIEWS, spinFactor,
} from "./state.js";
import { makeTube, updateTube } from "./trail.js";
import { makeBall, makeGhost, updateBall, spinBall } from "./ball.js";
import { renderHud, buildStrip } from "./hud.js";
import { makePitcher, poseAtRelease } from "./pitcher.js";
import { buildPark } from "./park/index.js";
import { initLook, setTimeOfDay, refreshShadows, inkProof, timeOfDay, TIMES } from "./look.js";

// No MSAA: the composer renders to its own targets, and SMAA (look.js) anti-aliases instead.
const renderer = new THREE.WebGLRenderer({ antialias: false });
renderer.setPixelRatio(window.devicePixelRatio);
document.body.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.add(buildPark()); // swappable: see park/index.js
scene.add(buildField());
// The ball is tiny on screen: an outline would swamp it, and no line may cross it (inkProof).
const ball = inkProof(makeBall());
const ghost = inkProof(makeGhost());
scene.add(ball, ghost);
// Movement aids (trail.js): a fading trail behind each flying ball, and the dashed no-spin path.
const trail = makeTube({ color: 0xffe07a, radiusFt: 0.035, fadeS: 0.12 });
const ghostTrail = makeTube({ color: 0xffb070, radiusFt: 0.03, fadeS: 0.12, opacity: 0.7 });
const noSpinLine = makeTube({ color: 0xffffff, radiusFt: 0.022, dashS: 0.012, opacity: 0.85 });
scene.add(trail, ghostTrail, noSpinLine);
let zone = null;
let figure = null; // the pitcher, rebuilt per pitcher (handedness), re-posed per pitch

// Vertical FOV chosen so every pitch's zone, plate arrival and release fit (see cameras.js).
// Far plane 6000 ft: the free camera can pull back to the whole floating island (~1300 ft across).
const camera = new THREE.PerspectiveCamera(CATCHER_VFOV_DEG, 1, 0.05, 6000);
camera.layers.enable(1); // layer 1: drawn but not outlined (look.js)
camera.layers.enable(2); // layer 2: not outlined and never inked over (look.js)
const look = initLook(renderer, scene, camera);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enabled = false;
controls.minDistance = FREE_MIN_DIST_FT;
controls.maxDistance = FREE_MAX_DIST_FT;
let freePose = { position: toScene(FREE_START.position), target: toScene(FREE_START.target) };
let appliedView = null;
const VFOV = {
  catcher: CATCHER_VFOV_DEG, batter: BATTER_EYE_VFOV_DEG, side: SIDE_VFOV_DEG, free: FREE_VFOV_DEG,
  ride: RIDE_VFOV_DEG,
};

/** Point the camera for the current view. Runs every frame: cheap, and ride follows t. */
function updateCamera() {
  const ab = currentAtBat(state);
  const path = shownPaths(state)[0]; // none for no_tracking: keep the last pose
  if (state.view !== appliedView) {
    // Leaving Free: remember the orbit (the other views overwrite the camera); entering: restore it.
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
  if (state.view === "catcher") pose = catcherView();
  else if (state.view === "batter") pose = batterEye(ab.stand);
  else if (path && state.view === "side") pose = sideView(path, camera.aspect);
  else if (path) pose = rideAlong(path, state.t, currentPitch(state).release, state.data.pitcher.throws);
  if (!pose) return;
  camera.position.copy(toScene(pose.position));
  camera.lookAt(toScene(pose.target));
}

function resize() {
  renderer.setSize(window.innerWidth, window.innerHeight);
  look.setSize(window.innerWidth, window.innerHeight);
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
  if (ab !== shown.ab) buildStrip(state, (i) => dispatch({ type: "goPitch", index: i }));
  if (pitch !== shown.pitch) { // zone height is per pitch (Statcast sz_top / sz_bot)
    if (zone) scene.remove(zone);
    zone = pitch.sz_top_ft == null ? null : buildZone(pitch.sz_top_ft, pitch.sz_bot_ft);
    if (zone) scene.add(zone);
    // Each pitch has its own release and arm angle; a no_tracking row may have no release, so
    // the figure keeps the previous pose (and a missing arm angle keeps the previous one).
    if (pitch.release?.every(Number.isFinite)) {
      poseAtRelease(figure, pitch.release, pitch.arm_angle_deg ?? figure.userData.armAngleDeg);
      refreshShadows(); // the pitcher moved: redraw his shadow
    }
  }
  shown = { ab, pitch };
  [...$("atbats").children].forEach((b, i) => b.classList.toggle("on", i === state.abIndex));
  $("picker-label").textContent = `${state.data.pitcher.name.split(",")[0]} · ${ab.game_date} · AB ${state.abIndex + 1}/${state.data.at_bats.length}`;
  renderHud(state, pitch);
}

const randomSeed = () => Math.floor(Math.random() * 2 ** 31);

async function pickPitcher(slug) {
  const data = await loadPitcher(slug);
  if (figure) {
    scene.remove(figure);
    figure.traverse((o) => {
      o.geometry?.dispose();
      for (const m of [o.material ?? []].flat()) m.dispose();
    });
  }
  figure = makePitcher(data.pitcher.throws);
  refreshShadows(); // new figure: redraw even if this at-bat's first pitch has no release
  scene.add(figure);
  for (const b of $("pitchers").children) b.classList.toggle("on", b.dataset.slug === slug);
  // ponytail: a plain list; a search box if at-bat lists grow past a few hundred.
  $("atbats").replaceChildren(...data.at_bats.map((ab, i) => {
    const b = document.createElement("button");
    // No outcome in the label: it would spoil the at-bat. (The pitch count is on the strip anyway.)
    b.innerHTML = `${ab.game_date} · ${ab.top ? "Top" : "Bot"} ${ab.inning} · vs ${ab.stand}HB <span class="sub">· ${ab.pitches.length} pitches</span>`;
    b.onclick = () => { dispatch({ type: "pickAtBat", index: i }); togglePicker(false); };
    return b;
  }));
  $("ab-count").textContent = `(${data.at_bats.length} held-out)`;
  dispatch({ type: "pickPitcher", slug, data });
  dispatch({ type: "pickAtBat", index: 0 });
}

/** Open or close the pitcher / at-bat picker (no argument: toggle). */
function togglePicker(open = $("picker").hidden) {
  $("picker").hidden = !open;
  if (open) $("atbats").children[state.abIndex]?.scrollIntoView({ block: "center" });
}

// Every action has a key and a button (headless screenshots cannot press keys).
const KEYS = {
  Space: { type: "replay" }, ArrowRight: { type: "nextPitch" }, ArrowLeft: { type: "prevPitch" },
  KeyR: { type: "setMode", mode: "real" }, KeyS: { type: "setMode", mode: "sim" },
  KeyG: { type: "setMode", mode: "ghost" },
  // 1-5: Catcher, Batter, Side, Ride, Free (the order of VIEWS and of the tabs).
  ...Object.fromEntries(VIEWS.map((view, i) => [`Digit${i + 1}`, { type: "setView", view }])),
  KeyN: { type: "toggleNoSpin" },
  KeyZ: { type: "setSpeed", speed: 1 }, KeyX: { type: "setSpeed", speed: 0.25 },
  KeyC: { type: "setSpeed", speed: 0.1 },
};
const TOD_NAMES = Object.keys(TIMES);
/** Next time of day: day -> sunset -> night -> day. */
const cycleTimeOfDay = () => {
  setTimeOfDay(TOD_NAMES[(TOD_NAMES.indexOf(timeOfDay()) + 1) % TOD_NAMES.length]);
  $("tod").textContent = `${timeOfDay()[0].toUpperCase()}${timeOfDay().slice(1)}`;
};
window.addEventListener("keydown", (e) => {
  // Leave browser shortcuts (Ctrl+R, ...) alone, and keys aimed at a focused control: a select
  // uses arrows itself, and Space on a focused button would also click it (a double replay).
  if (!state.data || e.ctrlKey || e.metaKey || e.altKey) return;
  if (["SELECT", "BUTTON", "SUMMARY"].includes(e.target.tagName)) return;
  if (e.code === "KeyA") return dispatch({ type: "pickAtBat", index: "random", seed: randomSeed() });
  if (e.code === "KeyT") return cycleTimeOfDay();
  if (e.code === "KeyP") return togglePicker();
  if (e.code === "Escape") return togglePicker(false);
  if (e.key === "?") return void ($("help").open = !$("help").open);
  if (KEYS[e.code]) {
    e.preventDefault();
    dispatch(KEYS[e.code]);
  }
});
// Drop focus after a click or a pick so the keys work again straight away.
document.addEventListener("click", (e) => { e.target.closest("button, summary")?.blur(); });
$("prev").onclick = () => dispatch({ type: "prevPitch" });
$("replay").onclick = () => dispatch({ type: "replay" });
$("next").onclick = () => dispatch({ type: "nextPitch" });
$("random").onclick = () => {
  dispatch({ type: "pickAtBat", index: "random", seed: randomSeed() });
  togglePicker(false);
};
$("picker-btn").onclick = () => togglePicker();
$("nospin").onclick = () => dispatch({ type: "toggleNoSpin" });
for (const m of MODES) $(`mode-${m}`).onclick = () => dispatch({ type: "setMode", mode: m });
for (const v of VIEWS) $(`view-${v}`).onclick = () => dispatch({ type: "setView", view: v });
for (const v of SPEEDS) $(`speed-${v}`).onclick = () => dispatch({ type: "setSpeed", speed: v });
$("tod").onclick = cycleTimeOfDay;

// Initial state from the URL, e.g. ?pitcher=sale-chris&ab=0&pitch=2&t=0.3&mode=ghost&nospin=1
// (frozen at t, for screenshots); ab=random&seed=N picks a seeded random at-bat.
const q = new URLSearchParams(location.search);
// ?tod=day|sunset|night (unknown names fall back to day).
setTimeOfDay(q.get("tod") ?? "day");
$("tod").textContent = `${timeOfDay()[0].toUpperCase()}${timeOfDay().slice(1)}`;

async function start() {
  const index = await loadIndex();
  $("pitchers").replaceChildren(...index.map((p) => {
    const b = document.createElement("button");
    b.dataset.slug = p.slug;
    b.innerHTML = `${p.name} <span class="sub">${p.throws}HP · ${p.n_at_bats} AB</span>`;
    b.onclick = () => pickPitcher(p.slug);
    return b;
  }));
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
  // ?cam=x,y,z,tx,ty,tz (Statcast ft): free-view camera position and target, for close-ups.
  const cam = (q.get("cam") ?? "").split(",").map(Number);
  if (cam.length === 6 && cam.every(Number.isFinite)) {
    freePose = { position: toScene(cam.slice(0, 3)), target: toScene(cam.slice(3)) };
  }
  if (q.has("view")) dispatch({ type: "setView", view: q.get("view") });
  if (q.has("speed")) dispatch({ type: "setSpeed", speed: Number(q.get("speed")) });
  if (q.get("nospin") === "1") dispatch({ type: "toggleNoSpin" });
  // ?panel=picker|help|nolook: open the picker or the key help, or fold the look card.
  if (q.get("panel") === "picker") togglePicker(true);
  if (q.get("panel") === "help") $("help").open = true;
  if (q.get("panel") === "nolook") $("look").open = false;
  // Freeze at t (no playback) so a headless screenshot catches the ball mid-flight.
  dispatch({ type: "seek", t: Number(q.get("t") ?? 0) });
  if (fpsBox) benchmark();
}
start();

// ?fps=1: frame-rate readout, averaged over half-second windows.
const fpsBox = q.get("fps") === "1" ? $("fps") : null;
if (fpsBox) fpsBox.hidden = false;
let fpsFrames = 0, fpsSince = null;
// Which GPU the browser gave WebGL (a software renderer such as SwiftShader is far slower).
const gl = renderer.getContext();
const gpu = gl.getExtension("WEBGL_debug_renderer_info");
const gpuName = gpu ? gl.getParameter(gpu.UNMASKED_RENDERER_WEBGL) : "unknown GPU";
function countFrame(now) {
  if (!fpsBox) return;
  fpsSince ??= now;
  fpsFrames++;
  if (now - fpsSince >= 500) {
    fpsBox.innerHTML = `${((fpsFrames * 1000) / (now - fpsSince)).toFixed(1)} fps · ${renderer.info.render.calls} draws · ${(renderer.info.render.triangles / 1e6).toFixed(2)} M tris<br>${gpuName}${burst}`;
    fpsFrames = 0;
    fpsSince = now;
  }
}

/**
 * Render-cost check for ?fps=1: 100 frames back to back, then a pixel read, which waits for the
 * GPU to finish them all. Gives ms per frame independent of requestAnimationFrame pacing (a
 * headless browser hardly runs rAF, so its live fps says nothing about the GPU).
 */
let burst = "";
function benchmark() {
  updateCamera();
  look.render(); // warm-up: shaders compiled, shadow map drawn
  const px = new Uint8Array(4);
  const t0 = performance.now();
  for (let i = 0; i < 100; i++) look.render();
  gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, px);
  const ms = (performance.now() - t0) / 100;
  const t1 = performance.now();
  refreshShadows();
  look.render(); // one frame that also redraws the shadow map (happens on each pitch change)
  gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, px);
  const shadowMs = performance.now() - t1;
  console.log(`render burst (${state.view}, ${timeOfDay()}): ${ms.toFixed(2)} ms/frame; with shadow redraw ${shadowMs.toFixed(1)} ms`);
  burst = `<br>burst: ${ms.toFixed(2)} ms/frame (${(1000 / ms).toFixed(0)} fps if GPU-bound)`;
}

let last = null;
renderer.info.autoReset = false; // count the whole frame (all passes), reset below
renderer.setAnimationLoop((now) => {
  countFrame(now);
  renderer.info.reset();
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
    updateTube(trail, state.mode === "sim" ? p.sim : p.real, state.t);
    updateTube(ghostTrail, state.mode === "ghost" ? p.sim : null, state.t);
    updateTube(noSpinLine, state.showNoSpin ? p.nospin : null, state.t);
  }
  if (state.data) updateCamera();
  look.render();
});
