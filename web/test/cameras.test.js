import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync, readdirSync, existsSync } from "node:fs";
import { PerspectiveCamera } from "three";
import {
  batterEye, BATTER_EYE_VFOV_DEG, umpireView, sideView, rideAlong, SIDE_VFOV_DEG, RIDE_VFOV_DEG, introSweep, INTRO_S, tvView,
} from "../src/cameras.js";
import {
  fenceDistFt, FENCE_HEIGHT_FT, BATTERS_EYE_GAP_FT, BATTERS_EYE_THICK_FT, BATTERS_EYE_HEIGHT_FT, BATTERS_EYE_HALF_DEG,
} from "../src/park/layout.js";
import { stateAt, visibleEnd } from "../src/trajectory.js";
import { BALL_RADIUS_FT } from "../src/ball.js";
import { toScene } from "../src/coords.js";

test("a right-handed batter stands on the catcher's left (x < 0), a lefty on the right", () => {
  assert.ok(batterEye("R").position[0] < 0);
  assert.ok(batterEye("L").position[0] > 0);
});

const here = new URL(".", import.meta.url);
const files = [new URL("fixtures/synthetic.json", here)];
const dataDir = new URL("../public/data/", here);
if (existsSync(dataDir)) {
  for (const f of readdirSync(dataDir)) {
    if (f.endsWith(".json") && f !== "index.json") files.push(new URL(f, dataDir));
  }
}

const flight = { t: [0, 0.2, 0.4], p: [[-1.5, 54, 6], [-1, 27, 4.2], [0, 1.4, 2.5]] };
const MARGIN = 0.95; // inside 95% of the frame in normalised device coordinates
const HALF_PLATE_FT = 17 / 24;
const PLATE_FRONT_Y_FT = 17 / 12;

/** A Three.js camera for a {position, target} pose (Statcast ft) at 1600x900. */
function cameraAt(pose, vfovDeg) {
  const cam = new PerspectiveCamera(vfovDeg, 1600 / 900, 0.05, 2000);
  cam.position.copy(toScene(pose.position));
  cam.lookAt(toScene(pose.target));
  cam.updateMatrixWorld();
  return cam;
}

/**
 * Points of one pitch that must be in frame (Ryan's rule, 2026-11-11): release and zone corners
 * always; for each shown path (Statcast fit, simulated, and with `noSpin` the no-spin path) the
 * end of its visible flight: the plate arrival, or where the ball reaches the ground (centre at one
 * radius, where ball.js hides it) if it gets there first. Paths are only seen where the ball is.
 */
function framingPoints(p, noSpin) {
  const pts = [p.release];
  for (const path of [p.real, p.sim, noSpin ? p.nospin : null]) {
    if (path) pts.push(visibleEnd(path, BALL_RADIUS_FT).p);
  }
  for (const z of [p.sz_bot_ft, p.sz_top_ft]) {
    for (const x of [-HALF_PLATE_FT, HALF_PLATE_FT]) pts.push([x, PLATE_FRONT_Y_FT, z]);
  }
  return pts;
}

/**
 * Every exported pitch's release, visible path ends (Statcast fit, simulated, and with `noSpin` the
 * no-spin path) and zone corners must project in front of the camera and inside 95% of the frame.
 * cams: {R, L} camera per batter side. Returns [pitches checked, tightest margin in NDC].
 */
function checkFraming(cams, noSpin) {
  let pitches = 0;
  let tightest = Infinity; // smallest distance to the 95% border, in NDC
  for (const file of files) {
    for (const ab of JSON.parse(readFileSync(file, "utf8")).at_bats) {
      for (const p of ab.pitches) {
        if (!p.real) continue; // no_tracking: nothing is drawn
        for (const q of framingPoints(p, noSpin)) {
          const v = toScene(q);
          const inFront = v.clone().applyMatrix4(cams[ab.stand].matrixWorldInverse).z < 0;
          v.project(cams[ab.stand]);
          const room = MARGIN - Math.max(Math.abs(v.x), Math.abs(v.y));
          tightest = Math.min(tightest, room);
          assert.ok(inFront && room >= 0, `${file.pathname} ${ab.id} #${p.pitch_number} [${q}] out of frame`);
        }
        pitches++;
      }
    }
  }
  assert.ok(pitches > 0);
  return [pitches, tightest];
}

test("a path in the dirt is framed at its ground contact, not at its below-ground plate arrival", () => {
  const r = BALL_RADIUS_FT;
  const dirt = { t: [0, 0.2, 0.4], p: [[-1.5, 54, 6], [-1, 20, 1 + r], [0, PLATE_FRONT_Y_FT, -0.5]] };
  const end = visibleEnd(dirt, r);
  // From z = 1 + r to -0.5: one radius is reached 1/(1.5 + r) of the way along the last segment.
  const f = 1 / (1.5 + r);
  assert.ok(Math.abs(end.p[2] - r) < 1e-12);
  assert.ok(Math.abs(end.t - (0.2 + 0.2 * f)) < 1e-12);
  assert.ok(Math.abs(end.p[1] - (20 + f * (PLATE_FRONT_Y_FT - 20))) < 1e-9);
  const pitch = { release: dirt.p[0], real: dirt, sim: null, nospin: dirt, sz_bot_ft: 1.6, sz_top_ft: 3.4 };
  const pts = framingPoints(pitch, true);
  assert.deepEqual(pts[1], end.p);
  assert.ok(!pts.some((q) => q[2] < 0), "the below-ground plate arrival must not be checked");
  // A path that stays up is framed at its plate arrival.
  assert.deepEqual(visibleEnd(flight, r).p, flight.p[2]);
});

test("batter's eye at 1600x900 frames every pitch's zone, visible path ends and release", () => {
  const cam = (stand) => cameraAt(batterEye(stand), BATTER_EYE_VFOV_DEG);
  const [n, tightest] = checkFraming({ R: cam("R"), L: cam("L") }, false);
  console.log(`batter framing: ${n} pitches checked, tightest margin ${tightest.toFixed(4)} NDC inside the 95% border`);
});

test("umpire's view is centred behind the plate at the umpire's spot, aim -15 deg, vfov 44 deg", () => {
  const v = umpireView();
  assert.deepEqual(v.position, [0, -5.0, 4.75]);
  assert.equal(v.target[0], 0);
  const d = v.target.map((x, i) => x - v.position[i]);
  assert.ok(Math.abs(Math.atan2(d[2], d[1]) * 180 / Math.PI + 15) < 1e-9);
  assert.equal(v.vfovDeg, 44);
});

test("umpire's view at 1600x900 frames every pitch's zone, visible path ends (incl. no spin) and release", () => {
  const v = umpireView();
  const cam = cameraAt(v, v.vfovDeg); // the same for both batter sides
  const [n, tightest] = checkFraming({ R: cam, L: cam }, true);
  console.log(`umpire framing: ${n} pitches checked, vfov ${v.vfovDeg} deg, tightest margin ${tightest.toFixed(4)} NDC inside the 95% border`);
});

// ---- Task 6: side view and ride-along

test("side view sits on the first-base side (x > 0) with release and plate in frame", () => {
  const v = sideView(flight);
  assert.ok(v.position[0] > 0);
  const cam = new PerspectiveCamera(SIDE_VFOV_DEG, 16 / 9, 0.05, 2000);
  cam.position.copy(toScene(v.position));
  cam.lookAt(toScene(v.target));
  cam.updateMatrixWorld();
  for (const q of [flight.p[0], flight.p[2]]) {
    const n = toScene(q).project(cam);
    assert.ok(Math.abs(n.x) < 1 && Math.abs(n.y) < 1, `${q} out of frame (${n.x}, ${n.y})`);
  }
});

test("ride starts over the throwing shoulder: behind, above and outside the release point", () => {
  for (const throws of ["R", "L"]) {
    const release = throws === "R" ? [-1.5, 54, 6] : [1.5, 54, 6];
    const path = { t: flight.t, p: flight.p.map(([x, y, z]) => [throws === "R" ? x : -x, y, z]) };
    const v = rideAlong(path, 0, release, throws);
    assert.ok(v.position[1] > release[1] && v.position[2] > release[2]);
    // Throwing side: a righty's arm is on the catcher's left (x < 0), a lefty's on the right.
    assert.ok(throws === "R" ? v.position[0] < release[0] : v.position[0] > release[0]);
  }
});

test("ride at the end of the flight has the zone in frame", () => {
  for (const end of [[0, 1.42, 2.5], [1.8, 1.42, 0.3], [-1.5, 1.42, 4.2]]) {
    const path = { t: flight.t, p: [flight.p[0], flight.p[1], end] };
    const cam = cameraAt(rideAlong(path, 0.4, path.p[0], "R"), RIDE_VFOV_DEG);
    for (const x of [-HALF_PLATE_FT, HALF_PLATE_FT]) {
      for (const z of [1.5, 3.5]) {
        const v = toScene([x, PLATE_FRONT_Y_FT, z]);
        const inFront = v.clone().applyMatrix4(cam.matrixWorldInverse).z < 0;
        v.project(cam);
        assert.ok(inFront && Math.abs(v.x) < 1 && Math.abs(v.y) < 1, `zone corner [${x}, ${z}] out of frame for end ${end}`);
      }
    }
  }
});

test("ride-along at mid-flight is behind (larger y) and above the ball, looking down the flight", () => {
  const t = 0.2;
  const ball = stateAt(flight, t).p;
  const v = rideAlong(flight, t, flight.p[0], "R");
  assert.ok(v.position[1] > ball[1]);
  assert.ok(v.position[2] > ball[2]);
  assert.ok(v.target[1] < ball[1]); // looks toward the plate
});

// ---- Task 9: intro sweep

test("intro sweep ends exactly at the umpire's view (and holds there after the end)", () => {
  const u = umpireView();
  for (const t of [INTRO_S, INTRO_S + 5]) {
    const v = introSweep(t);
    for (let i = 0; i < 3; i++) {
      assert.ok(Math.abs(v.position[i] - u.position[i]) < 1e-9);
      assert.ok(Math.abs(v.target[i] - u.target[i]) < 1e-9);
    }
    assert.equal(v.vfovDeg, u.vfovDeg);
  }
});

test("intro sweep starts out over the whole island and only ever closes in on the plate", () => {
  const start = introSweep(0).position;
  assert.ok(Math.hypot(start[0], start[1]) > 1000 && start[2] > 400, `${start}`);
  let prev = Infinity;
  for (let t = 0; t <= INTRO_S; t += INTRO_S / 400) {
    const d = Math.hypot(...introSweep(t).position);
    assert.ok(d <= prev + 1e-9, `moved away from the plate at t = ${t}`);
    prev = d;
  }
});

test("intro sweep clears the stands: over them (55-175 ft from home) it stays above roof or net", () => {
  for (let t = 0; t <= INTRO_S; t += INTRO_S / 2000) {
    const [x, y, z] = introSweep(t).position;
    const r = Math.hypot(x, y);
    // Roof and upper deck top out at 93 ft (100 to 160 ft out); lower bowl and backstop net, 32 ft.
    if (r > 100 && r < 175) assert.ok(z > 100, `t = ${t}: ${z.toFixed(1)} ft up at ${r.toFixed(0)} ft`);
    else if (r > 55 && r <= 100) assert.ok(z > 40, `t = ${t}: ${z.toFixed(1)} ft up at ${r.toFixed(0)} ft`);
  }
});

// ---- TV broadcast view (centre-field camera)

const TV_RUBBER_Y_FT = 60.5;
const MOUND_TOP_FT = 10 / 12;

test("TV view frames the pitcher on the rubber, his release, the zone and the plate at 1600x900", () => {
  const v = tvView();
  const cam = cameraAt(v, v.vfovDeg);
  const pts = [
    [0, 0, 0], [-HALF_PLATE_FT, 0, 0], [HALF_PLATE_FT, 0, 0], // home plate
    [0, TV_RUBBER_Y_FT, MOUND_TOP_FT], [0, TV_RUBBER_Y_FT, MOUND_TOP_FT + 6.5], // standing figure on the rubber
    [-2, TV_RUBBER_Y_FT, MOUND_TOP_FT + 6.5], [2, TV_RUBBER_Y_FT, MOUND_TOP_FT], // arms out to the sides
    [-2.5, 55, 5], [2.5, 55, 6.5], [0, 54, 6], // release points
  ];
  for (const z of [1.2, 3.8]) for (const x of [-HALF_PLATE_FT, HALF_PLATE_FT]) pts.push([x, PLATE_FRONT_Y_FT, z]);
  for (const q of pts) {
    const w = toScene(q);
    const inFront = w.clone().applyMatrix4(cam.matrixWorldInverse).z < 0;
    w.project(cam);
    assert.ok(inFront && Math.abs(w.x) < MARGIN && Math.abs(w.y) < MARGIN, `[${q}] out of frame (${w.x}, ${w.y})`);
  }
  assert.ok(v.vfovDeg < 8, "a telephoto lens: a few degrees");
});

test("TV camera is behind the pitcher, 25-40 ft up and off to the third-base side (x < 0), as on the broadcast", () => {
  const v = tvView();
  assert.ok(v.position[1] > TV_RUBBER_Y_FT + 300);
  assert.ok(v.position[2] >= 25 && v.position[2] <= 40, `${v.position[2]} ft up`);
  // Broadcast frames show the pitcher low-left with the plate up and to his right: a camera on
  // the third-base side (left-centre), 7-42 ft off the centre line depending on the park.
  assert.ok(v.position[0] <= -7 && v.position[0] >= -42, `${v.position[0]} ft across`);
});

test("TV frame layout matches the broadcast: pitcher low-left, plate just below mid-frame and right of him", () => {
  const v = tvView();
  const cam = cameraAt(v, v.vfovDeg);
  // Image fractions (0,0 top-left), as measured on two centre-field frames of Ohtani (Dodger
  // Stadium, Houston): feet 0.76/0.85 down, plate 0.55/0.50 down, plate 0.20/0.03 W right of him.
  const at = (q) => { const w = toScene(q).project(cam); return [(w.x + 1) / 2, (1 - w.y) / 2]; };
  const feet = at([0, TV_RUBBER_Y_FT, MOUND_TOP_FT]);
  const head = at([0, TV_RUBBER_Y_FT, MOUND_TOP_FT + 6.3]);
  const plate = at([0, PLATE_FRONT_Y_FT / 2, 0]);
  assert.ok(feet[1] > 0.74 && feet[1] < 0.87, `feet ${feet[1]}`);
  assert.ok(feet[1] - head[1] > 0.34 && feet[1] - head[1] < 0.44, `pitcher height ${feet[1] - head[1]} of the frame`);
  assert.ok(plate[1] > 0.48 && plate[1] < 0.6, `plate ${plate[1]}`);
  assert.ok(plate[0] - feet[0] > 0.03 && plate[0] - feet[0] < 0.2, `plate ${plate[0] - feet[0]} W right of the pitcher`);
});

test("TV line of sight to home plate and to the release clears the fence, the batter's eye wall and the stands", () => {
  const cam = tvView().position;
  for (const end of [[0, 0, 2.5], [0, 0, 0], [-1.5, 54, 6], [1.5, 54, 6]]) {
    for (let i = 0; i <= 2000; i++) {
      const f = i / 2000;
      const [x, y, z] = cam.map((c, k) => c + f * (end[k] - c));
      const angle = (Math.atan2(x, y) * 180) / Math.PI;
      const r = Math.hypot(x, y);
      // Bleachers and stands start outside the batter's eye wedge: the line never leaves it.
      assert.ok(Math.abs(angle) < BATTERS_EYE_HALF_DEG, `leaves the centre-field wedge at ${angle} deg`);
      const fence = fenceDistFt(angle);
      const eyeIn = fence + BATTERS_EYE_GAP_FT;
      if (r >= fence - 0.5 && r <= fence + 0.5) assert.ok(z > FENCE_HEIGHT_FT, `inside the fence: ${z} ft`);
      if (r >= eyeIn && r <= eyeIn + BATTERS_EYE_THICK_FT) assert.ok(z > BATTERS_EYE_HEIGHT_FT, `inside the batter's eye wall: ${z} ft`);
    }
  }
});

test("TV near plane gives depth resolution at the pitcher under 0.01 ft", () => {
  const v = tvView();
  const d = Math.hypot(...v.position.map((c, i) => c - [0, 60.5, 5][i]));
  assert.ok(v.near > 0.05);
  assert.ok(d * d / (v.near * 2 ** 24) < 0.01, `resolution ${d * d / (v.near * 2 ** 24)} ft`);
});

// ---- ride windup

const PRE_S = 2.85;
// Pelvis walks 5 ft toward the plate over the windup (stand-in for the delivery's pelvis joint).
const pelvisAt = (t) => [-0.5, 61 - 5 * Math.min(1, (t + PRE_S) / PRE_S), 3.8];
const windup = (t) => ({ pelvis: pelvisAt(t) });
const wpath = { t: flight.t, p: flight.p };

test("windup ride sits behind the pitcher on his throwing side with head and feet in 16:9 frame", () => {
  for (const throws of ["R", "L"]) {
    const m = throws === "R" ? 1 : -1;
    const path = { t: flight.t, p: flight.p.map(([x, y, z]) => [m * x, y, z]) };
    const t = -1.5, [px, py, pz] = pelvisAt(t);
    const w = { pelvis: [m * px, py, pz] };
    const v = rideAlong(path, t, path.p[0], throws, w);
    assert.ok(v.position[1] > py);
    assert.ok(throws === "R" ? v.position[0] < w.pelvis[0] : v.position[0] > w.pelvis[0]);
    const cam = cameraAt(v, RIDE_VFOV_DEG);
    for (const q of [[w.pelvis[0], py, 6.6], [w.pelvis[0], py, 0.3]]) {
      const n = toScene(q).project(cam);
      assert.ok(Math.abs(n.x) < 0.9 && Math.abs(n.y) < 0.9, `${q} out of frame (${n.x}, ${n.y})`);
    }
  }
});

test("windup ride is continuous into release and the flight, and equals today's pose at t = 0", () => {
  const dt = 1 / 240;
  let prev = null;
  for (let t = -PRE_S; t <= 0.3 + 1e-9; t += dt) {
    const v = rideAlong(wpath, t, wpath.p[0], "R", windup(t));
    if (prev) {
      // Camera speed: the glide covers ~5 ft in 0.4 s; smootherstep peaks at 1.875x the mean (~23 ft/s).
      // 0.15 ft per 1/240 s = 36 ft/s allows that but no jump.
      // (after release the camera rides the ball at ~130 ft/s, so only t <= 0 is bounded)
      assert.ok(t > 0 || Math.hypot(...v.position.map((c, i) => c - prev.position[i])) < 0.15, `position jump at t=${t}`);
      const a = v.target.map((c, i) => c - v.position[i]), b = prev.target.map((c, i) => c - prev.position[i]);
      const ang = Math.acos(Math.min(1, a.reduce((s, c, i) => s + c * b[i], 0) / Math.hypot(...a) / Math.hypot(...b)));
      assert.ok(ang < 0.01, `aim jump ${ang} rad at t=${t}`); // 0.57 deg per 1/240 s = 137 deg/s
    }
    prev = v;
  }
  const a = rideAlong(wpath, 0, wpath.p[0], "R", windup(0)), b = rideAlong(wpath, 0, wpath.p[0], "R");
  for (let i = 0; i < 3; i++) {
    assert.ok(Math.abs(a.position[i] - b.position[i]) < 1e-6 && Math.abs(a.target[i] - b.target[i]) < 1e-6);
  }
});

test("ride without a windup is the same for every t < 0 as at t = 0", () => {
  assert.deepEqual(rideAlong(wpath, -1, wpath.p[0], "R"), rideAlong(wpath, 0, wpath.p[0], "R"));
});
