import { test } from "node:test";
import assert from "node:assert/strict";
import { isOld } from "../src/park/old.js";
import {
  rpmLabel, spinAngleRad, wallBays, palmSites, inSector, clearForSignature, FALLS, WALL, GRAND, WALL_TOP_FT, TOWER, TOWER_V1, scoreboardDeg,
} from "../src/park/signature.js";
import { cliffHeight } from "../src/park/falls.js";
import { voussoirs } from "../src/park/grand.js";
import { SURFACE_KINDS, SURFACE_GLSL } from "../src/surface.js";
import { fenceDistFt, polar } from "../src/park/layout.js";
import { umpireView, UMPIRE_VFOV_DEG } from "../src/cameras.js";

test("old flags: tower, falls and oracle each switch off on their own", () => {
  assert.ok(isOld("tower", "?old=tower,falls"));
  assert.ok(isOld("falls", "?old=tower,falls"));
  assert.ok(!isOld("oracle", "?old=tower,falls"));
  assert.ok(isOld("oracle", "?old=oracle"));
});

test("rpmLabel: rounded, thousands separator, dashes without data", () => {
  assert.equal(rpmLabel(2277.4), "2,277");
  assert.equal(rpmLabel(2277.6), "2,278");
  assert.equal(rpmLabel(980), "980");
  assert.equal(rpmLabel(null), "---");
  assert.equal(rpmLabel(undefined), "---");
  assert.equal(rpmLabel(NaN), "---");
});

test("spinAngleRad: rpm / 60 turns per second, times t, times the slow-motion factor", () => {
  assert.ok(Math.abs(spinAngleRad(1800, 1) - 2 * Math.PI * 30) < 1e-9);
  assert.ok(Math.abs(spinAngleRad(2400, 0.4, 1 / 20) - 2 * Math.PI * 40 * 0.4 / 20) < 1e-9);
  assert.equal(spinAngleRad(2400, 0), 0);
  assert.ok(Number.isNaN(spinAngleRad(null, 1)) || spinAngleRad(null, 1) === 0); // no spin data: spinBall hides the rod, the angle is moot
});

test("the Spin Tower's ball fits the band of sky the umpire sees above the batter's eye wall", () => {
  const u = umpireView();
  const d = TOWER.y - u.position[1];
  const eye = u.position[2];
  const deg = (z) => (Math.atan2(z - eye, d) * 180) / Math.PI;
  // Top of the umpire's frame: the aim pitch plus half the vertical field of view.
  const pitch = Math.atan2(u.target[2] - u.position[2], u.target[1] - u.position[1]);
  const topDeg = (pitch * 180) / Math.PI + UMPIRE_VFOV_DEG / 2;
  assert.ok(deg(TOWER.ballZ + TOWER.ballR) < topDeg - 0.2, `ball top ${deg(TOWER.ballZ + TOWER.ballR)} vs frame top ${topDeg}`);
  const wallTopDeg = (Math.atan2(36 - eye, 426 + 5) * 180) / Math.PI; // the eye wall: 36 ft tall, 426 ft from home
  assert.ok(deg(TOWER.ballZ - TOWER.ballR) > wallTopDeg, "ball bottom clears the batter's eye wall");
  // The boards beside the ball (crest, readout) sit in the same band, and the tower is nearer than the first-built 690 ft.
  assert.ok(deg(TOWER.ballZ + TOWER.boardH / 2) < topDeg - 0.2 && deg(TOWER.ballZ - TOWER.boardH / 2) > wallTopDeg);
  assert.ok(TOWER.y < TOWER_V1.y && TOWER.y > 426 + 26 + 3 + 20, "closer than before, still behind the batter's eye wall");
  // The batter's eye stays plain dark: no readout on its wall any more (only ?old=tower paints one).
  assert.ok(Math.abs(TOWER.x) + TOWER.boardX + TOWER.boardW / 2 < 160);
});

test("scoreboard: dry ground beside the falls (left of the pool at 22 deg), as first built under ?old=falls", () => {
  const legDeg = (24 / (fenceDistFt(scoreboardDeg()) + 80)) * (180 / Math.PI); // its legs stand 24 ft off the centre, ~80 ft behind the fence
  assert.ok(scoreboardDeg() + legDeg < FALLS.a0, "both legs are on the bleachers' side of the pool");
  assert.ok(scoreboardDeg() - legDeg > 9 + 2, "and clear of the batter's eye");
});

test("wall bays: continuous from the right-field bleachers to the foul pole and round the corner, on the fence line", () => {
  const all = wallBays();
  const bays = all.filter((b) => !b.ret);
  assert.ok(bays.length >= 7 && bays.length <= 14, `${bays.length} bays`);
  assert.equal(all.length - bays.length, 1, "one return across the bullpen gap");
  const ret = all.find((b) => b.ret);
  assert.ok(ret.len > 25 && ret.len < 35);
  for (const b of bays.slice(0, -2)) { // the last two turn the corner into foul ground
    const a = (Math.atan2(b.p[0], b.p[1]) * 180) / Math.PI;
    assert.ok(a >= FALLS.a0 - 1 && a <= FALLS.a1 + 1, `bearing ${a}`);
    assert.ok(Math.abs(Math.hypot(...b.p) - (fenceDistFt(a) + FALLS.wallOff)) < 2, "on the fence line");
  }
  for (const b of bays) assert.ok(b.len > 8 && b.len <= WALL.bay * 1.05, `bay ${b.len}`);
  for (let i = 1; i < bays.length; i++) assert.ok(Math.hypot(bays[i].p[0] - bays[i - 1].p[0], bays[i].p[1] - bays[i - 1].p[1]) < WALL.bay * 1.1, "no gaps");
  const first = bays[0].p, a0 = polar(FALLS.a0, fenceDistFt(FALLS.a0) + FALLS.wallOff);
  assert.ok(Math.hypot(first[0] - a0[0], first[1] - a0[1]) < WALL.bay, "starts at the bleachers");
});

test("palms stand beyond left field, behind the left bleachers' back wall", () => {
  const palms = palmSites();
  assert.ok(palms.length >= 12);
  for (const s of palms) {
    const a = (Math.atan2(s.p[0], s.p[1]) * 180) / Math.PI;
    assert.ok(a < -9 && a > -50);
    assert.ok(Math.hypot(...s.p) > fenceDistFt(a) + 71 + 10);
  }
});

test("cliff: flat at its foot, a steep face rising to the top, lower at the sides", () => {
  assert.equal(cliffHeight(FALLS.aFall, -3), 0);
  assert.equal(cliffHeight(FALLS.aFall, 0), 0);
  const top = cliffHeight(FALLS.aFall, FALLS.faceRun);
  assert.ok(top > 60 && top <= FALLS.cliffH);
  assert.ok(cliffHeight(FALLS.aFall, 20) < top);
  assert.ok(cliffHeight(FALLS.aCliff0, FALLS.faceRun) < 5);
});

test("clearForSignature: keeps scatter out of the pool, the cliff and the tower, and out of the palms; leaves the rest", () => {
  const inFalls = polar(FALLS.aFall, fenceDistFt(FALLS.aFall) + 150);
  const behindBall = [10, 560];
  const elsewhere = [-200, 20];
  const plan = {
    trees: [{ p: inFalls }, { p: behindBall }, { p: elsewhere }], groundTrees: [{ p: inFalls }, { p: elsewhere }],
    bushes: [], rocks: [], lamps: [{ p: inFalls }], benches: [], tents: [{ p: inFalls }], paths: [[inFalls, [0, 0]], [[-100, 100], [-90, 200]]],
  };
  clearForSignature(plan);
  assert.deepEqual(plan.trees.map((t) => t.p), [elsewhere]);
  assert.equal(plan.groundTrees.length, 1);
  assert.equal(plan.lamps.length, 0);
  assert.equal(plan.tents.length, 0);
  assert.equal(plan.paths.length, 1);
  assert.ok(inSector(inFalls, FALLS.a0, FALLS.a1, 100));
});

test("grand wall: taller than the first-built one, arch fits inside the arcade with room for the keystone and the frieze", () => {
  assert.ok(GRAND.h1 + GRAND.h2 > WALL_TOP_FT + 8, "monumental: at least 8 ft above the old wall");
  assert.ok(GRAND.parapetTop > GRAND.cornice && GRAND.pedestalTop > GRAND.parapetTop && GRAND.lampTop > GRAND.pedestalTop);
  const crown = GRAND.spring + GRAND.openW / 2; // top of the opening, ft above the arcade's base
  assert.ok(crown + 2.2 + 1.0 < GRAND.h2 - 2.5, "keystone clears the frieze band");
  assert.ok(GRAND.openW < WALL.bay - 4, "a pier at least 4 ft wide between openings");
  assert.ok(GRAND.sill < GRAND.spring);
});

test("voussoirs: nine stones round the half circle, the middle one (keystone) bigger", () => {
  const ring = voussoirs(7, 2.2, 4, 0, 9);
  assert.equal(ring.length, 9);
  const reach = (geo) => { geo.computeBoundingBox(); return geo.boundingBox.max.y; };
  assert.ok(reach(ring[4]) > reach(ring[3]) && reach(ring[4]) > reach(ring[5]));
});

test("brick surface: kind 6 with a running-bond shader that fades by pixel footprint", () => {
  assert.equal(SURFACE_KINDS.brick, 6);
  assert.ok(SURFACE_GLSL.includes("brickSurface") && SURFACE_GLSL.includes("mod(row, 2.0)"));
  assert.ok(SURFACE_GLSL.includes("smoothstep(0.18, 0.5, fp)"));
});
