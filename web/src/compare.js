/**
 * Eye-test page (M5 Task 8): a real broadcast clip from Ryan's disk next to the replica, synced on the
 * release frame, with the replica's skeleton projected onto the video and the clip's 2D score.
 *
 * The clip is opened with <input type="file"> and played from an object URL: it never touches the server
 * (CLAUDE.md: video is personal research, never served or committed).
 *
 * Overlay: scripts/m5_compare2d.py fits each clip ONE affine camera A (2x3) plus a per-frame pan b, so the
 * delivery's joints (Statcast ft) land on the video as image_px = A p + b[frame]. The overlay draws the
 * DELIVERY's joints (what the score measures, without the per-pitch hand pin); the 3D figure on the right
 * is the viewer's rig (fixed bone lengths, feet on the mound, hand pinned to this pitch's release), so the
 * two differ slightly near release by design.
 */
import * as THREE from "three";
import { toScene } from "./coords.js";
import { buildField } from "./field.js";
import { tvView } from "./cameras.js";
import { loadDelivery } from "./data.js";
import { loadHandModels } from "./handModels.js";
import { showOpenHand } from "./figure.js";
import { boneLengths } from "./pitcher.js";
import { showNoWebGL, showNoCompare } from "./notices.js";
import { pitcherApi } from "./signed_off/index.js"; // Ohtani keeps his signed-off pose code

// ---- pure helpers (tested in test/compare.test.js) ---------------------------------------------------

/** The clips.json key for a picked file ("<game_pk>_<ab>_<pitch>.mp4"; a browser may prefix a fake path). */
export function clipKey(filename) {
  const base = String(filename).split(/[\\/]/).pop();
  return /^\d+_\d+_\d+\.mp4$/.test(base) ? base : null;
}

/** Replica time (s from release, t = 0) for a video time: release is at `releaseFrame` of a clip at `fps`. */
export function replicaTime(videoTime, releaseFrame, fps) {
  return videoTime - releaseFrame / fps;
}

/**
 * The frame on screen at `videoTime` (frame f shows from f/fps to (f+1)/fps), plus a manual `offset`
 * in frames. Why the knob: this assumes the mp4's first frame is at t = 0; a file with a start offset
 * would put the overlay a constant few frames off, and [ / ] fix that by eye.
 */
export function videoFrame(videoTime, fps, offset = 0) {
  return Math.floor(videoTime * fps + 1e-6) + offset;
}

/** Image point [x, y] px of a Statcast-ft point p through the fitted camera: A p + b. */
export function projectPoint(A, b, p) {
  return A.map((row, i) => row[0] * p[0] + row[1] * p[1] + row[2] * p[2] + b[i]);
}

/** The fitted pan of a clip at a video frame, or null (outside the window, or a frame that was not fit). */
export function panAt(clip, frame) {
  return clip.b[frame - clip.frame0] ?? null;
}

/** Name of the phase holding t (phases are (t0, t1]; the first also holds its t0), or null outside. */
export function phaseAt(phases, t) {
  if (!phases.length || t < phases[0].t0_s || t > phases[phases.length - 1].t1_s) return null;
  return phases.find((p) => t <= p.t1_s).name;
}

/** The delivery's joints `idx` at t s from release, linear between its frames; null outside the delivery. */
export function jointsAt(d, t, idx) {
  const k = d.release_index + t * d.fps;
  if (k < 0 || k > d.joints_ft.length - 1) return null;
  const i0 = Math.floor(k), i1 = Math.min(i0 + 1, d.joints_ft.length - 1), a = k - i0;
  return idx.map((j) => d.joints_ft[i0][j].map((v, c) => (1 - a) * v + a * d.joints_ft[i1][j][c]));
}

// ---- the page ------------------------------------------------------------------------------------------

// Bones of the 12-joint skeleton, in compare.json joint order (l/r shoulder, elbow, wrist, hip, knee, ankle).
const EDGES = [[0, 1], [0, 2], [2, 4], [1, 3], [3, 5], [0, 6], [1, 7], [6, 7], [6, 8], [8, 10], [7, 9], [9, 11]];
const PHASE_LABEL = {
  set_rocker: "Set / rocker", leg_lift: "Leg lift", stride: "Stride", arm_action: "Arm action / release",
  follow_through: "Follow-through",
};

/** Loading feedback: a spinner in the status line; a failure shows the message with Retry. */
async function main() {
  const slug = new URLSearchParams(location.search).get("slug") ?? "ohtani-shohei";
  const status = document.getElementById("status");
  const initial = (main.prompt ??= status.textContent); // the page's own prompt, kept across Retry
  status.replaceChildren(Object.assign(document.createElement("span"), { className: "spin" }), `Loading ${slug}'s delivery…`);
  try {
    await run(slug);
  } catch (err) {
    console.error("compare page failed to load:", err);
    if (/webgl|context/i.test(String(err?.message))) return showNoWebGL(); // the renderer constructor threw
    const retry = Object.assign(document.createElement("button"), { textContent: "Retry", onclick: () => main() });
    const back = Object.assign(document.createElement("a"), { textContent: "Back to the park", href: "/" });
    status.replaceChildren(`Couldn't load ${slug}'s delivery. `, retry, " ", back);
    return;
  }
  if (status.querySelector(".spin")) status.textContent = initial; // loaded: back to the prompt
}

async function run(slug) {
  const $ = (id) => document.getElementById(id);
  const [delivery, cmp] = await Promise.all([
    loadDelivery(slug),
    fetch(`/data/deliveries/${slug}.compare.json`).then((r) => (r.ok ? r.json() : null)).catch(() => null),
    loadHandModels(), // sculpted glove/hand before makePitcher (?models=0: procedural)
  ]);
  if (!delivery || !cmp) {
    $("status").textContent = "No video comparison for this pitcher yet.";
    showNoCompare(slug); // visitor card; the developer hint goes to the console
    return;
  }

  // 3D replica, from the centre-field TV camera (cameras.js), the same preset as the viewer's TV view.
  const pane = $("replica");
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  pane.appendChild(renderer.domElement);
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x3f7a3a); // outfield grass behind the mound, as on TV
  scene.add(new THREE.HemisphereLight(0xffffff, 0x6b5a40, 1.6), buildField());
  const sun = new THREE.DirectionalLight(0xffffff, 1.6);
  sun.position.set(-50, 120, 80);
  scene.add(sun);
  const tv = tvView();
  const camera = new THREE.PerspectiveCamera(tv.vfovDeg, 1, tv.near, 6000);
  camera.position.copy(toScene(tv.position));
  camera.lookAt(toScene(tv.target));
  const api = pitcherApi(slug);
  const figure = api.makePitcher(delivery.throws, delivery.height_in ?? 75, boneLengths(delivery));
  scene.add(figure);
  const resize = () => {
    const w = pane.clientWidth, h = pane.clientHeight;
    renderer.setSize(w, h, false); // false: CSS sizes the canvas, so it never pushes the pane wider
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    shown = null;
  };
  new ResizeObserver(resize).observe(pane);

  const video = $("video"), canvas = $("overlay"), ctx = canvas.getContext("2d");
  let clip = null, key = null, offset = 0, shown = null;

  $("file").addEventListener("change", () => {
    const file = $("file").files[0];
    if (!file) return;
    key = clipKey(file.name);
    clip = key ? cmp.clips[key] ?? null : null;
    if (!clip) {
      if (video.src) URL.revokeObjectURL(video.src); // don't leak the previous pick's blob URL
      video.removeAttribute("src");
      video.load();
      $("status").textContent = `${file.name}: clip not in this pitcher's index`;
      $("scores").replaceChildren();
      $("split").textContent = "";
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      return;
    }
    if (video.src) URL.revokeObjectURL(video.src);
    video.src = URL.createObjectURL(file); // local file -> blob: URL; nothing is uploaded or fetched
    offset = 0;
    shown = null;
    setZoom();
    $("status").textContent = key;
    $("split").textContent = clip.split === "test" ? "TEST (held-out game)" : "TRAIN (fed the replica)";
    $("split").className = `split ${clip.split}`;
    renderScores();
  });
  video.addEventListener("loadedmetadata", () => {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    $("stage").style.aspectRatio = `${video.videoWidth} / ${video.videoHeight}`;
    // start a little before the delivery window, like the clip itself
    video.currentTime = (clip.frame0 + 0.5) / clip.fps;
  });

  const fmt = (v) => `${v.toFixed(3)} <small>(~${(v * cmp.torso_in).toFixed(1)} in)</small>`;
  function renderScores() {
    const rows = [["Overall", clip.score.overall, clip.floor_2d.overall, clip.naive_static.overall, null]]
      .concat(cmp.phases.map((p) => [PHASE_LABEL[p.name], clip.score.phases[p.name], clip.floor_2d.phases[p.name],
        clip.naive_static.phases[p.name], p.name]));
    $("scores").innerHTML = `<tr><th></th><th>Replica</th><th>His other pitches</th><th>Static pose</th></tr>` +
      rows.map(([label, r, f, n, name]) => `<tr data-phase="${name ?? ""}"><th>${label}</th>` +
        [r, f, n].map((v) => `<td>${Number.isFinite(v) ? fmt(v) : "–"}</td>`).join("") + "</tr>").join("");
  }

  function drawOverlay(frame, t) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (!$("show-overlay").checked) return;
    const b = panAt(clip, frame), pts = b && jointsAt(delivery, t, cmp.smpl_idx);
    if (!pts) return;
    const px = pts.map((p) => projectPoint(clip.A, b, p));
    ctx.lineWidth = Math.max(2, canvas.height / 240);
    ctx.strokeStyle = "rgba(255, 60, 60, 0.9)";
    ctx.beginPath();
    for (const [i, j] of EDGES) { ctx.moveTo(...px[i]); ctx.lineTo(...px[j]); }
    ctx.stroke();
    ctx.fillStyle = "#fff";
    for (const [x, y] of px) { ctx.beginPath(); ctx.arc(x, y, ctx.lineWidth * 1.3, 0, 2 * Math.PI); ctx.fill(); }
  }

  function tick() {
    requestAnimationFrame(tick);
    if (!clip || !video.src) return;
    // One frame number drives both sides, so the replica and the overlay sit on the frame on screen.
    const frame = videoFrame(video.currentTime, clip.fps, offset);
    const state = `${frame}|${$("show-overlay").checked}`;
    if (state === shown) return;
    shown = state;
    const t = (frame - clip.release_pose_frame) / clip.fps; // = replicaTime at that frame's start
    api.poseAtTime(figure, delivery, t, clip.release_ft, clip.arm_angle_deg ?? 35);
    showOpenHand(figure, t >= 0); // hand opens once the ball has left it
    renderer.render(scene, camera);
    drawOverlay(frame, t);
    const phase = phaseAt(cmp.phases, t);
    for (const tr of $("scores").querySelectorAll("tr")) tr.classList.toggle("now", !!phase && tr.dataset.phase === phase);
    $("clock").textContent = `t = ${t >= 0 ? "+" : ""}${t.toFixed(3)} s · frame ${frame}` +
      (offset ? ` (offset ${offset > 0 ? "+" : ""}${offset})` : "") + (phase ? ` · ${PHASE_LABEL[phase]}` : "");
  }
  requestAnimationFrame(tick);

  // ---- controls ----
  const step = (n) => {
    if (!clip) return;
    video.pause();
    const f = videoFrame(video.currentTime, clip.fps) + n;
    video.currentTime = (Math.max(f, 0) + 0.5) / clip.fps; // mid-frame: robust to float rounding
  };
  const toggle = () => (video.paused ? video.play() : video.pause());
  $("play").addEventListener("click", toggle);
  video.addEventListener("play", () => { $("play").textContent = "Pause"; });
  video.addEventListener("pause", () => { $("play").textContent = "Play"; });
  for (const btn of document.querySelectorAll("[data-speed]")) {
    btn.addEventListener("click", () => {
      video.playbackRate = Number(btn.dataset.speed);
      for (const o of document.querySelectorAll("[data-speed]")) o.classList.toggle("on", o === btn);
    });
  }
  $("show-overlay").addEventListener("change", () => { shown = null; });
  // Zoom: 2x about the pitcher's median image position over the clip (his projected hip-mid), fixed per
  // clip so the picture does not shake with the per-frame fit. Why 2x: he is ~0.4 of the frame tall, so
  // 2x keeps his whole stride in view.
  function setZoom() {
    const z = $("zoom");
    if (!clip || !$("zoom-on").checked) { z.style.transform = ""; return; }
    const xs = [], ys = [];
    clip.b.forEach((b, i) => {
      const pts = b && jointsAt(delivery, (clip.frame0 + i - clip.release_pose_frame) / clip.fps, cmp.smpl_idx);
      if (!pts) return;
      const [x, y] = projectPoint(clip.A, b, pts[6].map((v, c) => (v + pts[7][c]) / 2));
      xs.push(x); ys.push(y);
    });
    const med = (a) => a.sort((p, q) => p - q)[a.length >> 1];
    // scale 2 about the top-left, then shift his position to the middle (kept inside the picture)
    const shift = (c, size) => Math.min(0, Math.max(-100, 50 - (200 * c) / size));
    z.style.transformOrigin = "0 0";
    z.style.transform = `translate(${shift(med(xs), canvas.width)}%, ${shift(med(ys), canvas.height)}%) scale(2)`;
  }
  $("zoom-on").addEventListener("change", setZoom);
  video.addEventListener("loadedmetadata", setZoom);
  window.addEventListener("keydown", (e) => {
    if (e.target.tagName === "INPUT" && e.target.type !== "checkbox") return;
    if (e.key === "ArrowLeft") step(-1);
    else if (e.key === "ArrowRight") step(1);
    else if (e.key === " ") toggle();
    else if (e.key === "o") $("show-overlay").click();
    else if (e.key === "z") $("zoom-on").click();
    else if (e.key === "[" || e.key === "]") { offset += e.key === "]" ? 1 : -1; shown = null; }
    else return;
    e.preventDefault();
  });
}

if (typeof document !== "undefined") main(); // node tests import the helpers only
