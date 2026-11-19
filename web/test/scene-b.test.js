import { test } from "node:test";
import assert from "node:assert/strict";
import { isOld } from "../src/park/old.js";
import { planFans, occupancy, awayShare } from "../src/park/crowd.js";
import { adLayout, adUV, AD_BRANDS, AD_ROWS } from "../src/park/home.js";
import { tvView } from "../src/cameras.js";
import { WALL_H_FT } from "../src/park/stands.js";

test("isOld: comma list in ?old=", () => {
  assert.ok(isOld("crowd", "?old=crowd,home"));
  assert.ok(isOld("home", "?x=1&old=crowd,home"));
  assert.ok(!isOld("structure", "?old=crowd,home"));
  assert.ok(!isOld("crowd", ""));
});

// A synthetic stand (test data, not Statcast): 3 sections of 14 seats, 10 rows, on each side of home.
const seats = [];
for (const deck of ["lower", "upper"]) for (const sx of [-1, 1]) for (let row = 0; row < 10; row++) for (let i = 0; i < 42; i++) {
  seats.push({ p: [sx * (80 + i * 2), 60 + row], h: row, yaw: 0, deck, row, rows: 10, sec: Math.floor(i / 14) });
}

test("crowd: home colours are the majority, away fans cluster in pockets, standing fans exist", () => {
  const fans = planFans(seats, 11).filter(Boolean);
  const share = (t) => fans.filter((f) => f.team === t).length / fans.length;
  assert.ok(share("home") > 0.6, `home ${share("home")}`);
  assert.ok(share("away") > 0.05 && share("away") < 0.3, `away ${share("away")}`);
  assert.ok(fans.some((f) => f.standing) && fans.some((f) => f.armsUp));
  // The visitor pocket is the lower deck on the third-base side (x < 0), not the first-base side.
  const lower = (sx) => planFans(seats, 11).map((f, i) => [f, seats[i]]).filter(([f, s]) => f && s.deck === "lower" && Math.sign(s.p[0]) === sx);
  const awayFrac = (l) => l.filter(([f]) => f.team === "away").length / l.length;
  assert.ok(awayFrac(lower(-1)) > 3 * awayFrac(lower(1)));
});

test("crowd: denser behind home than in the far corners; same seed, same crowd", () => {
  const near = { p: [0, -60], deck: "lower", row: 0, rows: 18 }, far = { p: [250, 230], deck: "lower", row: 0, rows: 18 };
  assert.ok(occupancy(near) > occupancy(far) + 0.2);
  assert.ok(awayShare({ p: [-150, 100], deck: "lower" }, 0.9) > 0.5);
  assert.deepEqual(planFans(seats, 5), planFans(seats, 5));
});

test("home ads: boards on the 60 ft wall arc facing home, 4:1, within the 3.4 ft wall, one row; neighbours differ", () => {
  const boards = adLayout();
  assert.equal(AD_ROWS.length, 1);
  assert.ok(boards.length >= 9);
  for (const b of boards) {
    assert.ok(Math.abs(Math.hypot(...b.p) - 59.7) < 0.01);
    assert.ok(b.p[1] < 0 && b.n[1] > 0.5 || Math.abs(b.p[0]) > 10, "faces home");
    assert.ok(Math.abs(b.w / b.h - 4) < 1e-9 && b.z + b.h / 2 <= WALL_H_FT);
    assert.ok(b.tile >= 0 && b.tile < AD_BRANDS.length);
  }
  const at = (row, k) => boards.filter((b) => b.z === AD_ROWS[row].z)[k];
  const n = boards.length / AD_ROWS.length;
  for (let row = 0; row < AD_ROWS.length; row++) for (let k = 0; k < n; k++) {
    if (k + 1 < n) assert.notEqual(at(row, k).tile, at(row, k + 1).tile);
    if (row + 1 < AD_ROWS.length) for (const d of [-1, 0, 1]) if (at(row + 1, k + d)) assert.notEqual(at(row, k).tile, at(row + 1, k + d).tile);
  }
});

test("home ads: atlas tiles tile the unit square; the TV camera's wall strip is covered by boards", () => {
  for (let i = 0; i < AD_BRANDS.length; i++) {
    const [u0, v0, u1, v1] = adUV(i);
    assert.ok(u0 >= 0 && u1 <= 1 && v0 >= 0 && v1 <= 1 && u1 > u0 && v1 > v0);
  }
  // The board directly behind home (centre of the arc) is within a few ft of the TV camera's line to the plate.
  const tv = tvView().position;
  const ahead = adLayout().filter((b) => Math.abs(b.p[0] - (-tv[0] * 60) / tv[1]) < 8);
  assert.ok(ahead.length >= 1, "boards in the TV frame");
});
