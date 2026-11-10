import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync, readdirSync, existsSync } from "node:fs";
import { PerspectiveCamera } from "three";
import {
  batterEye, BATTER_EYE_VFOV_DEG, catcherView, sideView, rideAlong, SIDE_VFOV_DEG, RIDE_VFOV_DEG,
} from "../src/cameras.js";
import { stateAt } from "../src/trajectory.js";
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
 * Every exported pitch's release, plate arrivals (Statcast fit, simulated, and with `noSpin` the
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
        const pts = [p.release, [p.plate_real[0], PLATE_FRONT_Y_FT, p.plate_real[1]]];
        if (p.plate_sim) pts.push([p.plate_sim[0], PLATE_FRONT_Y_FT, p.plate_sim[1]]);
        if (noSpin && p.plate_nospin) pts.push([p.plate_nospin[0], PLATE_FRONT_Y_FT, p.plate_nospin[1]]);
        for (const z of [p.sz_bot_ft, p.sz_top_ft]) {
          for (const x of [-HALF_PLATE_FT, HALF_PLATE_FT]) pts.push([x, PLATE_FRONT_Y_FT, z]);
        }
        for (const q of pts) {
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

test("batter's eye at 1600x900 frames every pitch's zone, plate arrival and release", () => {
  const cam = (stand) => cameraAt(batterEye(stand), BATTER_EYE_VFOV_DEG);
  const [n, tightest] = checkFraming({ R: cam("R"), L: cam("L") }, false);
  console.log(`batter framing: ${n} pitches checked, tightest margin ${tightest.toFixed(4)} NDC inside the 95% border`);
});

test("catcher's view is centred behind the plate", () => {
  const v = catcherView();
  assert.equal(v.position[0], 0);
  assert.equal(v.target[0], 0);
  assert.ok(v.position[1] < 0); // behind the back point of the plate
});

test("catcher's view at 1600x900 frames every pitch's zone, plate arrivals (incl. no spin) and release", () => {
  const v = catcherView();
  const cam = cameraAt(v, v.vfovDeg); // the same for both batter sides
  const [n, tightest] = checkFraming({ R: cam, L: cam }, true);
  console.log(`catcher framing: ${n} pitches checked, vfov ${v.vfovDeg} deg, tightest margin ${tightest.toFixed(4)} NDC inside the 95% border`);
});

// ---- Task 6: side view and ride-along
const flight = { t: [0, 0.2, 0.4], p: [[-1.5, 54, 6], [-1, 27, 4.2], [0, 1.4, 2.5]] };

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
