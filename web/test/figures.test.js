import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync, readdirSync, existsSync } from "node:fs";
import { Vector3 } from "three";
import { JOINTS, J } from "../src/skeleton.js";
import { castJoints, castVisibility, buildCast, setCast } from "../src/figures.js";
import { UMPIRE_Y_FT, UMPIRE_EYE_HEIGHT_FT } from "../src/cameras.js";
import { BALL_RADIUS_FT } from "../src/ball.js";
import { toScene } from "../src/coords.js";

const HALF_PLATE_FT = 17 / 24;
const PLATE_FRONT_Y_FT = 17 / 12;
const TORSO = new Set(["pelvis", "spine1", "spine2", "spine3", "neck", "head"].map((n) => J[n]));

/** Body capsules {a, b, r} (Statcast ft): every bone, torso and head fat, plus the bat. */
function capsules(kind, stand) {
  const { joints, bat } = castJoints(kind, stand);
  const caps = JOINTS.flatMap((j, i) => (j.parent < 0 ? [] : [{ a: joints[j.parent], b: joints[i], r: TORSO.has(j.parent) ? 0.55 : 0.3 }]));
  if (bat) caps.push({ a: bat.knob, b: bat.tip, r: 0.11 });
  return caps;
}
function segDist(p, { a, b }) {
  const ab = b.map((v, i) => v - a[i]);
  const t = Math.max(0, Math.min(1, ab.reduce((s, v, i) => s + v * (p[i] - a[i]), 0) / ab.reduce((s, v) => s + v * v, 0)));
  return Math.hypot(...p.map((v, i) => v - (a[i] + t * ab[i])));
}
const all = (kind, stand) => castJoints(kind, stand).joints;

for (const stand of ["R", "L"]) {
  const side = stand === "L" ? 1 : -1;
  test(`${stand}HB stands in his own box (|x| 1.2-5.2 ft on the ${side < 0 ? "third" : "first"}-base side), facing the plate`, () => {
    const p = all("batter", stand);
    for (const n of ["pelvis", "left_ankle", "right_ankle", "left_foot", "right_foot"]) {
      const [x, y] = p[J[n]];
      assert.ok(side * x > 1.2 && side * x < 5.2 && y > -2.3 && y < 3.7, `${n} at ${x}, ${y}`);
    }
    // Facing the plate: his toes point toward it; his front (lead) shoulder toward the pitcher.
    assert.ok(side * (p[J.left_foot][0] - p[J.left_ankle][0]) < 0);
    const lead = stand === "L" ? "right" : "left";
    assert.ok(p[J[`${lead}_shoulder`]][1] > p[J.pelvis][1]);
    assert.ok(p[J.head][2] > 5 && p[J.head][2] < 6.5, `head ${p[J.head][2]} ft up`);
  });

  test(`catcher squats centred behind the plate, the umpire stands behind him in the slot (${stand}HB)`, () => {
    const c = all("catcher", stand);
    const u = all("umpire", stand);
    assert.ok(c.every((q) => q[1] < 0), "catcher wholly behind the plate's back point");
    assert.ok(Math.abs(c[J.pelvis][0]) < 0.3 && c[J.head][2] < 4, `catcher pelvis x ${c[J.pelvis][0]}, head ${c[J.head][2]}`);
    assert.ok(u[J.pelvis][1] < c[J.pelvis][1] - 2);
    // Slot: toward the batter's side; head at the umpire camera's spot.
    assert.ok(side * u[J.head][0] > 0.5);
    assert.ok(Math.abs(u[J.head][1] - UMPIRE_Y_FT) < 0.6, `umpire head y ${u[J.head][1]}`);
    assert.ok(Math.abs(u[J.head][2] - UMPIRE_EYE_HEIGHT_FT) < 0.5, `umpire head ${u[J.head][2]} ft up`);
  });

  test(`no figure enters the strike-zone volume over the plate (${stand}HB)`, () => {
    for (const kind of ["batter", "catcher", "umpire"]) {
      for (const cap of capsules(kind, stand)) {
        for (let f = 0; f <= 1; f += 0.05) {
          const [x, y, z] = cap.a.map((v, i) => v + f * (cap.b[i] - v));
          const inside = Math.abs(x) < HALF_PLATE_FT + cap.r && y > -cap.r && y < PLATE_FRONT_Y_FT + cap.r && z > 0.5 && z < 4.5;
          assert.ok(!inside, `${kind} at [${x.toFixed(2)}, ${y.toFixed(2)}, ${z.toFixed(2)}]`);
        }
      }
    }
  });
}

test("no figure is in the way of any pitch that reaches the plate within 1.5 ft of its centre", () => {
  const here = new URL(".", import.meta.url);
  const files = [new URL("fixtures/synthetic.json", here)];
  const dataDir = new URL("../public/data/", here);
  if (existsSync(dataDir)) for (const f of readdirSync(dataDir)) if (f.endsWith(".json") && f !== "index.json") files.push(new URL(f, dataDir));
  let n = 0;
  for (const file of files) {
    for (const ab of JSON.parse(readFileSync(file, "utf8")).at_bats ?? []) {
      const caps = ["batter", "catcher", "umpire"].flatMap((k) => capsules(k, ab.stand));
      for (const p of ab.pitches) {
        for (const path of [p.real, p.sim]) {
          // Inside pitches can hit the batter (a hit-by-pitch is real): only over-the-plate pitches count.
          if (!path || Math.abs(path.p.at(-1)[0]) > 1.5) continue;
          for (const q of path.p) for (const c of caps) assert.ok(segDist(q, c) > c.r + BALL_RADIUS_FT, `${ab.id} #${p.pitch_number} hits a figure at [${q}]`);
          n++;
        }
      }
    }
  }
  assert.ok(n > 0);
});

test("each eye view hides its own figure; the umpire's view also hides the catcher", () => {
  assert.deepEqual(castVisibility("umpire"), { batter: true, catcher: false, umpire: false });
  assert.deepEqual(castVisibility("batter"), { batter: false, catcher: true, umpire: true });
  for (const v of ["tv", "side", "ride", "free"]) assert.deepEqual(castVisibility(v), { batter: true, catcher: true, umpire: true });
});

test("the posed rig puts its joints on the pose (bones keep their lengths) and re-poses per stand", () => {
  const cast = buildCast();
  for (const stand of ["R", "L"]) {
    assert.equal(setCast(cast, stand, "tv"), true);
    assert.equal(setCast(cast, stand, "tv"), false, "nothing changed: no redraw");
    for (const kind of ["batter", "catcher", "umpire"]) {
      const fig = cast.userData[kind];
      const want = all(kind, stand);
      for (const n of ["head", "left_hand", "right_hand", "left_foot", "right_foot"]) {
        const got = fig.userData.bones[J[n]].getWorldPosition(new Vector3());
        assert.ok(got.distanceTo(toScene(want[J[n]])) < 0.08, `${kind} ${n} off by ${got.distanceTo(toScene(want[J[n]]))}`);
      }
    }
  }
  setCast(cast, "L", "umpire");
  assert.equal(cast.userData.umpire.visible, false);
  assert.equal(cast.userData.batter.visible, true);
});
