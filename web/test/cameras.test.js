import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync, readdirSync, existsSync } from "node:fs";
import { PerspectiveCamera } from "three";
import { batterEye, BATTER_EYE_VFOV_DEG } from "../src/cameras.js";
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

function cameraFor(stand) {
  const cam = new PerspectiveCamera(BATTER_EYE_VFOV_DEG, 1600 / 900, 0.05, 2000);
  const eye = batterEye(stand);
  cam.position.copy(toScene(eye.position));
  cam.lookAt(toScene(eye.target));
  cam.updateMatrixWorld();
  return cam;
}

test("batter's eye at 1600x900 frames every pitch's zone, plate arrival and release", () => {
  const cams = { R: cameraFor("R"), L: cameraFor("L") };
  let pitches = 0;
  let tightest = Infinity; // smallest distance to the 95% border, in NDC
  for (const file of files) {
    for (const ab of JSON.parse(readFileSync(file, "utf8")).at_bats) {
      for (const p of ab.pitches) {
        if (!p.real) continue; // no_tracking: nothing is drawn
        const pts = [p.release, [p.plate_real[0], PLATE_FRONT_Y_FT, p.plate_real[1]]];
        if (p.plate_sim) pts.push([p.plate_sim[0], PLATE_FRONT_Y_FT, p.plate_sim[1]]);
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
  console.log(`framing: ${pitches} pitches checked, tightest margin ${tightest.toFixed(4)} NDC inside the 95% border`);
  assert.ok(pitches > 0);
});
