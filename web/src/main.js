import "./ui.css";
import * as THREE from "three";
import { toScene } from "./coords.js";
import { buildField, buildZone, RUBBER_FRONT_Y_FT } from "./field.js";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import {
  batterEye, umpireView, tvView, sideView, rideAlong, BATTER_EYE_VFOV_DEG, UMPIRE_VFOV_DEG, SIDE_VFOV_DEG,
  RIDE_VFOV_DEG, FREE_VFOV_DEG, TV_VFOV_DEG, FREE_START, FREE_MIN_DIST_FT, FREE_MAX_DIST_FT, introSweep, INTRO_S,
} from "./cameras.js";
import { loadIndex, loadPitcher, loadDelivery } from "./data.js";
import {
  initialState, reduce, preroll, currentAtBat, currentPitch, shownPaths, MODES, SPEEDS, VIEWS, spinFactor,
} from "./state.js";
import { makeTube, updateTube } from "./trail.js";
import { makeBall, makeGhost, updateBall, spinBall } from "./ball.js";
import { renderHud, buildStrip } from "./hud.js";
import { makePitcher, poseAtRelease, poseAtTime, handPosition } from "./pitcher.js";
import { buildPark } from "./park/index.js";
import { initLook, setTimeOfDay, refreshShadows, inkProof, timeOfDay, TIMES } from "./look.js";

// No MSAA: the composer renders to its own targets, and SMAA (look.js) anti-aliases instead.
const renderer = new THREE.WebGLRenderer({ antialias: false });
renderer.setPixelRatio(window.devicePixelRatio);
document.body.appendChild(renderer.domElement);

const scene = new THREE.Scene();
const park = buildPark(); // swappable: see park/index.js
scene.add(park);
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
const camera = new THREE.PerspectiveCamera(UMPIRE_VFOV_DEG, 1, 0.05, 6000);
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
  umpire: UMPIRE_VFOV_DEG, batter: BATTER_EYE_VFOV_DEG, side: SIDE_VFOV_DEG, free: FREE_VFOV_DEG,
  ride: RIDE_VFOV_DEG, tv: TV_VFOV_DEG,
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
  if (state.view === "umpire") pose = umpireView();
  else if (state.view === "tv") pose = tvView();
  else if (state.view === "batter") pose = batterEye(ab.stand);
  // With a delivery the windup happens on the rubber, behind release: keep the pitcher in shot.
  else if (path && state.view === "side") pose = sideView(path, camera.aspect, state.delivery ? RUBBER_FRONT_Y_FT + 1.5 : null);
  else if (path) pose = rideAlong(path, Math.max(state.t, 0), currentPitch(state).release, state.data.pitcher.throws);
  if (!pose) return;
  camera.position.copy(toScene(pose.position));
  camera.lookAt(toScene(pose.target));
}

/*
 * Intro sweep (cameras.js introSweep): plays on load and on I, from the whole island down to the
 * umpire's view; any key, click or wheel skips it. Depth of field only here (look.setFocus), easing
 * to none by the end so the hand-over to the pitch view is sharp. The HUD panels fade out meanwhile.
 * intro = { t, frozen } while it runs, null otherwise.
 */
let intro = null;
function startIntro(t = 0, frozen = false) {
  // Leaving Free for the intro: endIntro clears appliedView, so save the orbit here.
  if (appliedView === "free") freePose = { position: camera.position.clone(), target: controls.target.clone() };
  if (state.data && state.view !== "umpire") dispatch({ type: "setView", view: "umpire" });
  intro = { t, frozen };
  controls.enabled = false;
  document.body.classList.add("intro");
}
function endIntro() {
  intro = null;
  look.setFocus(null);
  appliedView = null; // re-apply the view's FOV and controls
  document.body.classList.remove("intro");
}
/** Advance the intro by dt s and pose the camera; ends it after INTRO_S. */
function updateIntro(dt) {
  if (!intro.frozen) intro.t += dt;
  if (intro.t >= INTRO_S && !intro.frozen) return endIntro();
  const pose = introSweep(intro.t);
  camera.fov = pose.vfovDeg;
  camera.updateProjectionMatrix();
  camera.position.copy(toScene(pose.position));
  camera.lookAt(toScene(pose.target));
  const focus = Math.hypot(...pose.position.map((v, i) => v - pose.target[i]));
  look.setFocus(focus, 7 * (1 - Math.min(1, intro.t / INTRO_S)) ** 2);
}
for (const ev of ["pointerdown", "wheel"]) window.addEventListener(ev, () => { if (intro && !intro.frozen) endIntro(); });

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
    if (zone) { scene.remove(zone); zone.geometry.dispose(); zone.material.dispose(); }
    zone = pitch.sz_top_ft == null ? null : buildZone(pitch.sz_top_ft, pitch.sz_bot_ft);
    if (zone) scene.add(zone);
    // Each pitch has its own release and arm angle; a no_tracking row may have no release, so
    // the figure keeps the previous pose (and a missing arm angle keeps the previous one).
    // With a delivery the figure is posed every frame instead (render loop).
    if (!state.delivery && pitch.release?.every(Number.isFinite)) {
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
  const [data, delivery] = await Promise.all([loadPitcher(slug), loadDelivery(slug)]);
  if (figure) {
    scene.remove(figure);
    figure.traverse((o) => {
      o.geometry?.dispose();
      for (const m of [o.material ?? []].flat()) m.dispose();
      o.skeleton?.dispose(); // bone texture
    });
  }
  figure = makePitcher(data.pitcher.throws, data.pitcher.height_in ?? 75); // scaled to his height
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
  dispatch({ type: "pickPitcher", slug, data, delivery });
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
  // 1-6: Umpire, Batter, Side, Ride, Free, TV (the order of VIEWS and of the tabs).
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
  if (intro && !intro.frozen) { e.preventDefault(); return endIntro(); } // any key skips the intro
  if (e.code === "KeyI") return startIntro();
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

// Initial state from the URL, e.g. ?pitcher=sale-chris&ab=0&pitch=2&t=0.3&mode=ghost&nospin=1&view=umpire
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
  // The intro plays on load unless a view was asked for or ?intro=off; ?intro=<s> holds that frame
  // of it (for screenshots).
  const iq = q.get("intro");
  if (iq !== null && iq !== "off" && Number.isFinite(Number(iq))) startIntro(Number(iq), true);
  else if (iq === null && !q.has("view")) startIntro();
  if (q.has("speed")) dispatch({ type: "setSpeed", speed: Number(q.get("speed")) });
  if (q.get("nospin") === "1") dispatch({ type: "toggleNoSpin" });
  // ?panel=picker|help|nolook: open the picker or the key help, or fold the look card.
  if (q.get("panel") === "picker") togglePicker(true);
  if (q.get("panel") === "help") $("help").open = true;
  if (q.get("panel") === "nolook") $("look").open = false;
  // Freeze at t (no playback) so a headless screenshot catches the ball mid-flight.
  dispatch({ type: "seek", t: q.has("t") ? Number(q.get("t")) : 0 - preroll(state) }); // no ?t=: from the start of the delivery
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
  if (intro) updateIntro(0); else updateCamera();
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
  renderer.info.reset();
  look.render(); // one more, to count what a frame draws (all passes)
  const { calls, triangles } = renderer.info.render;
  console.log(`render burst (${state.view}, ${timeOfDay()}): ${ms.toFixed(2)} ms/frame; with shadow redraw ${shadowMs.toFixed(1)} ms; ${calls} draws, ${(triangles / 1e6).toFixed(2)} M tris`);
  burst = `<br>burst: ${ms.toFixed(2)} ms/frame (${(1000 / ms).toFixed(0)} fps if GPU-bound)`;
}

/** What the park's scoreboard shows: the count before the current pitch, and the pitch. */
function boardInfo() {
  const ab = currentAtBat(state), p = currentPitch(state);
  if (!p) return null;
  return {
    balls: p.balls, strikes: p.strikes, outs: ab.outs, inning: ab.inning, top: ab.top,
    pitcher: state.data.pitcher.name.split(",")[0], pitch: p.pitch_name ?? p.pitch_type, mph: p.speed_mph ?? null,
  };
}

/**
 * With a delivery (M5), pose the figure at the current t; the hand blends onto this pitch's
 * release just before t = 0. Re-posed (and its shadow redrawn) only when t or the pitch changed,
 * so a frozen frame costs nothing.
 */
let posedAt = null;
function poseFigure(p) {
  const key = `${state.t}|${state.abIndex}|${state.pitchIndex}`;
  if (!state.delivery || !figure || key === posedAt) return;
  posedAt = key;
  const release = p.release?.every(Number.isFinite) ? p.release : null;
  poseAtTime(figure, state.delivery, state.t, release, p.arm_angle_deg ?? figure.userData.armAngleDeg);
  refreshShadows();
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
    poseFigure(p);
    // Before release (t < 0) everything that follows the flight is hidden (it would clamp to the
    // release point); the ball rides in the hand instead. At t = 0 the hand is on the release
    // point, which is where the flight starts, so the hand-off is seamless.
    const flying = state.t >= 0;
    const ft = Math.max(state.t, 0);
    updateBall(ball, flying ? (state.mode === "sim" ? p.sim : p.real) : null, ft);
    if (!flying && figure) {
      ball.visible = true;
      ball.position.copy(handPosition(figure));
    }
    spinBall(ball, p.spin_rpm == null ? null : p.spin_axis_deg, 2 * Math.PI * (p.spin_rpm / 60) * ft * spinFactor(state.speed));
    updateBall(ghost, flying && state.mode === "ghost" ? p.sim : null, ft);
    updateTube(trail, flying ? (state.mode === "sim" ? p.sim : p.real) : null, ft);
    updateTube(ghostTrail, flying && state.mode === "ghost" ? p.sim : null, ft);
    updateTube(noSpinLine, flying && state.showNoSpin ? p.nospin : null, ft);
  }
  if (intro) updateIntro(dt);
  else if (state.data) updateCamera();
  park.userData.update?.({ timeS: now / 1000, board: boardInfo() });
  look.render();
});
