import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync, readdirSync, existsSync } from "node:fs";
import { PerspectiveCamera } from "three";
import {
  batterEye, BATTER_EYE_VFOV_DEG, umpireView, sideView, rideAlong, SIDE_VFOV_DEG, RIDE_VFOV_DEG, introSweep, INTRO_S,
} from "../src/cameras.js";
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
