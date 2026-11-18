// Ohtani's delivery is frozen to the viewer Ryan signed off (commit 70c9f22). The reference is
// rebuilt from git history at test time (git archive 70c9f22), not from a committed fixture: a
// fixture of joint positions would go stale silently if shared files (coords/field/look) drift,
// while the real 70c9f22 code, run beside the frozen module, catches that too.
import { test, before, after } from "node:test";
import assert from "node:assert/strict";
import { execSync } from "node:child_process";
import { existsSync, readFileSync, rmSync, mkdirSync } from "node:fs";
import { pathToFileURL, fileURLToPath } from "node:url";
import { Vector3 } from "three";
import * as frozen from "../src/signed_off/pitcher.js";
import * as current from "../src/pitcher.js";
import { SIGNED_OFF, pitcherApi } from "../src/signed_off/index.js";

const REF_DIR = fileURLToPath(new URL("./.ref-70c9f22/", import.meta.url)); // inside web/ so "three" resolves
let ref;
before(async () => {
  rmSync(REF_DIR, { recursive: true, force: true });
  mkdirSync(REF_DIR, { recursive: true });
  execSync(`git archive 70c9f22 ./src | tar -x -C test/.ref-70c9f22`, { cwd: fileURLToPath(new URL("..", import.meta.url)) });
  ref = await import(pathToFileURL(`${REF_DIR}src/pitcher.js`).href);
});
after(() => rmSync(REF_DIR, { recursive: true, force: true }));

const load = (rel) => { const u = new URL(rel, import.meta.url); return existsSync(u) ? JSON.parse(readFileSync(u)) : null; };
const bones = (fig) => {
  fig.updateMatrixWorld(true);
  const out = {};
  fig.traverse((o) => { if (o.name.startsWith("mixamorig:")) out[o.name] = o.getWorldPosition(new Vector3()); });
  return out;
};

function maxDeviation(delivery, releases, armAngle) {
  const a = ref.makePitcher(delivery.throws, delivery.height_in ?? 75);
  const b = frozen.makePitcher(delivery.throws, delivery.height_in ?? 75);
  const times = [];
  for (let t = -delivery.pre_s; t <= delivery.post_s; t += 1 / 60) times.push(t);
  times.push(-1.2345, -0.0171, 0.0123, 0.3333, 0.5);
  let worst = 0, n = 0;
  for (const release of releases) for (const t of times) {
    ref.poseAtTime(a, delivery, t, release, armAngle);
    frozen.poseAtTime(b, delivery, t, release, armAngle);
    const pa = bones(a), pb = bones(b);
    assert.ok(Object.keys(pa).length >= 24);
    for (const k in pa) { worst = Math.max(worst, pa[k].distanceTo(pb[k])); n++; }
    worst = Math.max(worst, ref.handPosition(a).distanceTo(frozen.handPosition(b)));
  }
  return { worst, n };
}

test("frozen pose path equals 70c9f22 on the synthetic delivery (every bone, within 1e-9 ft)", () => {
  const d = load("./fixtures/delivery-synthetic.json");
  const { worst } = maxDeviation(d, [[-2.4, 54.3, 5.7], [2.9, 54.2, 5.0]], 35);
  assert.ok(worst <= 1e-9, `max deviation ${worst}`);
});

test("frozen pose path equals 70c9f22 on Ohtani's real delivery and real releases", (t) => {
  const d = load("../public/data/deliveries/ohtani-shohei.json");
  const p = load("../public/data/ohtani-shohei.json");
  if (!d || !p) return t.skip("web/public/data not present");
  const pitches = p.at_bats.flatMap((ab) => ab.pitches).filter((x) => x.release?.every(Number.isFinite));
  const picks = [0, 0.25, 0.5, 0.75, 0.99].map((f) => pitches[Math.floor(f * pitches.length)]);
  for (const pk of picks) {
    const { worst, n } = maxDeviation(d, [pk.release], pk.arm_angle_deg ?? 35);
    console.log(`release ${pk.release}: ${n} bone positions, max deviation ${worst}`);
    assert.ok(worst <= 1e-9, `max deviation ${worst}`);
  }
});

test("pitcherApi: Ohtani gets the frozen path; everyone else the current one", () => {
  assert.deepEqual([...SIGNED_OFF], ["ohtani-shohei"]);
  assert.equal(pitcherApi("ohtani-shohei"), frozen);
  for (const s of ["misiorowski-jacob", "sale-chris", "king-michael", null]) assert.equal(pitcherApi(s), current);
});

test("current path builds figures from per-pitcher bone lengths (Misiorowski/Sale)", () => {
  const d = load("./fixtures/delivery-synthetic.json");
  const lens = current.boneLengths(d);
  const f = pitcherApi("sale-chris").makePitcher("L", 78, lens);
  assert.ok(f.userData.throws === "L" && Object.keys(bones(f)).length >= 24);
});
