// M7 "Face the pitcher": pure-JS at-bat engine over a face-1 JSON (no DOM).
// Python (scripts/m7_export.py) fits everything; this file only evaluates it and samples.
// Ports: LogisticSeq.predict, LocationModel.mean/sample, hitter.zone_bin/HitterModel/HitterTable,
// face.reaim_knots. Parity with Python is tested against face.parity in the fixture.
import { rng } from "../park/layout.js"; // mulberry32, already in the viewer

export { rng };

const HALF_W_FT = 17 / 24 + 1.45 / 12; // plate half-width + ball radius: any overlap counts as "in"
const BALL_R_FT = 1.45 / 12;
const BINS = ["heart", "edge_in", "edge_out", "chase", "waste"];
const BIN_EDGES_IN = [-3, 0, 3, 9];
const OUTCOMES = ["take", "whiff", "foul", "in_play"];

/** Context feature value by feature name (the exporter uses Python column names). */
const featValue = (name, ctx) => (name === "prev_type" ? ctx.prevType : ctx[name]);

/**
 * Sum of coef columns for the one-hot of `vals` (one string per feature). An unseen level one-hots
 * to all zeros (OneHotEncoder handle_unknown="ignore"), so it simply adds nothing.
 */
function oneHotDot(categories, vals, coefRow) {
  let off = 0, s = 0;
  categories.forEach((cats, f) => {
    const j = cats.indexOf(String(vals[f]));
    if (j >= 0) s += coefRow[off + j];
    off += cats.length;
  });
  return s;
}

const softmax = (z) => {
  const m = Math.max(...z), e = z.map((v) => Math.exp(v - m)), t = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / t);
};

/** P(next pitch type | count, stand, prevType, tto), over the bank types only. */
export function typeProbs(face, ctx) {
  const q = face.seq, cats = q.features.map((f) => q.categories[f]), vals = q.features.map((f) => featValue(f, ctx));
  const p = softmax(q.coef.map((row, k) => oneHotDot(cats, vals, row) + q.intercept[k]));
  const fl = p.map((v) => Math.max(v, q.floor)), t = fl.reduce((a, b) => a + b, 0); // floor, renormalise
  return Object.fromEntries(q.classes.map((c, k) => [c, fl[k] / t]));
}

/** Context-dependent mean plate location [x, z] ft of one pitch type (ridge mean shift). */
export function locationMean(face, type, ctx) {
  const l = face.location[type], cats = l.features.map((f) => l.categories[f]);
  const vals = l.features.map((f) => featValue(f, ctx));
  return l.coef.map((row, k) => oneHotDot(cats, vals, row) + l.intercept[k]);
}

const normal2 = (r) => { // Box-Muller: two independent standard normals
  const u = 1 - r(), v = r(), m = Math.sqrt(-2 * Math.log(u));
  return [m * Math.cos(2 * Math.PI * v), m * Math.sin(2 * Math.PI * v)];
};

/** Draw the final plate crossing [x, z] ft: mean + L z, L = Cholesky(cov). No extra scatter. */
export function sampleTarget(face, type, ctx, r) {
  const [[a, b], [, c]] = face.location[type].cov, m = locationMean(face, type, ctx);
  const l11 = Math.sqrt(a), l21 = b / l11, l22 = Math.sqrt(c - l21 * l21), [z1, z2] = normal2(r);
  return [m[0] + l11 * z1, m[1] + l21 * z1 + l22 * z2];
}

/** Signed distance (inches, negative inside) from the edge of the zone grown by one ball radius. */
function edgeIn(x, z, top, bot) {
  const dx = Math.abs(x) - HALF_W_FT, dz = Math.max(bot - BALL_R_FT - z, z - (top + BALL_R_FT));
  const d = dx < 0 && dz < 0 ? Math.max(dx, dz) : Math.hypot(Math.max(dx, 0), Math.max(dz, 0));
  return d * 12;
}

/** heart < -3, edge_in -3..0, edge_out 0..3, chase 3..9, waste > 9; edge value goes to the lower bin. */
export function zoneBin(x, z, top, bot) {
  const d = edgeIn(x, z, top, bot), i = BIN_EDGES_IN.findIndex((e) => d <= e); // np.digitize(right=True)
  return BINS[i < 0 ? BINS.length - 1 : i];
}

/** In the zone = signed edge distance <= 0 (same rule as the "edge_in"/"heart" bins). */
export function isStrike(face, stand, [x, z]) {
  const zn = face.zone[stand];
  return edgeIn(x, z, zn.top_ft, zn.bot_ft) <= 0;
}

/** {take, whiff, foul, in_play} for the stand's average hitter (logistic model or count x zone table). */
export function hitterProbs(face, stand, ctx, type, xz) {
  const h = face.hitter[stand], zn = face.zone[stand], bin = zoneBin(xz[0], xz[1], zn.top_ft, zn.bot_ft);
  let p;
  if (h.kind === "model") {
    const vals = { count: ctx.count, stand, pitch_type: type, zone_bin: bin };
    p = softmax(h.coef.map((row, k) => oneHotDot(h.categories, h.features.map((f) => vals[f]), row) + h.intercept[k]));
  } else {
    const c = h.cells[`${ctx.count}|${bin}`] ?? [0, 0, 0, 0], n = c.reduce((a, b) => a + b, 0);
    p = c.map((v, k) => (v + h.a * h.overall[k]) / (n + h.a));
  }
  return Object.fromEntries(OUTCOMES.map((o, k) => [o, p[k]]));
}

/**
 * Shift Hermite knots so the plate end moves by exactly (dx, dz) ft; release untouched.
 * u = (y0 - y)/(y0 - y_end) runs 0 -> 1, positions gain u*(dx, dz), velocities gain
 * (dx, dz)*(-vy)/(y0 - y_end) (du/dt) so the tangents match. Port of face.reaim_knots.
 */
export function reaim(sim, dx, dz) {
  const y0 = sim.s[0][1], y1 = sim.s.at(-1)[1];
  if (y0 === y1) throw new Error("reaim needs knots spanning a distance in y");
  const s = sim.s.map(([x, y, z, vx, vy, vz]) => {
    const u = (y0 - y) / (y0 - y1), k = -vy / (y0 - y1);
    return [x + u * dx, y, z + u * dz, vx + k * dx, vy, vz + k * dz];
  });
  return { s, T: sim.T };
}

/** Fresh at-bat (id is caller-supplied so the same seed gives the same at-bat). batter "none": every pitch is a take. tto: times through the order (string). */
export function newAtBat(face, { stand, batter = "avg", count = "0-0", tto = "1", id } = {}) {
  const [balls, strikes] = count.split("-").map(Number);
  return { id: id ?? "face-ab", stand, batter, tto, generated: true, pitches: [], balls, strikes, event: null };
}

/** Count after a result. foul with 2 strikes leaves it unchanged; whiff (incl. foul tip) is a strike. */
export function nextCount(ab, outcome, strike) {
  let { balls, strikes } = ab, event = null;
  const isStrikeCall = outcome === "whiff" || (outcome === "take" && strike);
  if (outcome === "in_play") event = "in_play";
  else if (outcome === "foul") { if (strikes < 2) strikes++; }
  else if (isStrikeCall) { if (++strikes >= 3) event = "strikeout"; }
  else if (++balls >= 4) event = "walk";
  return { balls, strikes, event };
}

/** Sample from a probability map with one uniform draw (never argmax). */
function draw(probs, r) {
  const u = r(), keys = Object.keys(probs);
  let c = 0;
  for (const k of keys) if (u < (c += probs[k])) return k;
  return keys.at(-1); // float round-off at u ~ 1
}

/** Throw one sampled pitch. Returns {ab (new), pitch}. Draw order: type, target, bank, outcome, tip. */
export function throwPitch(face, ab, r) {
  if (ab.event) throw new Error("at-bat is over");
  const count = `${ab.balls}-${ab.strikes}`, prev = ab.pitches.at(-1);
  const ctx = { count, stand: ab.stand, prevType: prev ? prev.pitch_type : "start", tto: ab.tto ?? "1" };
  const probs = typeProbs(face, ctx), type = draw(probs, r);
  const target = sampleTarget(face, type, ctx, r);
  const bank = face.bank[type], e = bank[Math.min(Math.floor(r() * bank.length), bank.length - 1)];
  const dx = target[0] - e.aim[0], dz = target[1] - e.aim[1];
  const strike = isStrike(face, ab.stand, target);

  let outcome = "take", description = strike ? "called_strike" : "ball";
  if (ab.batter !== "none") {
    outcome = draw(hitterProbs(face, ab.stand, ctx, type, target), r);
    if (outcome === "whiff") {
      // a caught foul tip is a strike that can end the AB; share measured on 2-strike whiffs
      description = ab.strikes === 2 && r() < face.hitter[ab.stand].foul_tip_share ? "foul_tip" : "swinging_strike";
    } else description = { take: description, foul: "foul", in_play: "hit_into_play" }[outcome];
  }
  const nc = nextCount(ab, outcome, strike);
  const pitch = {
    pitch_number: ab.pitches.length + 1, balls: ab.balls, strikes: ab.strikes,
    pitch_type: type, pitch_name: face.types[type].name, status: "ok",
    speed_mph: e.speed_mph, spin_rpm: e.spin_rpm, spin_axis_deg: e.spin_axis_deg,
    arm_angle_deg: face.types[type].arm_angle_deg, extension_ft: e.extension_ft, release: e.release,
    sz_top_ft: face.zone[ab.stand].top_ft, sz_bot_ft: face.zone[ab.stand].bot_ft,
    description, events: nc.event, real: null, generated: true,
    sim: reaim(e.sim, dx, dz), plate_sim: [e.plate[0] + dx, e.plate[1] + dz, e.plate[2]],
    // break is vs a no-spin flight at the bank aim; the re-aim keeps it (first-order shift)
    tunnel: { vaa_deg: null, haa_deg: null, perceived_mph: null, sep_prev_in: null,
      break_x_in: e.break_in[0], break_z_in: e.break_in[1] },
    outcome, why: { p: probs[type], count, stand: ab.stand },
  };
  return { ab: { ...ab, pitches: [...ab.pitches, pitch], balls: nc.balls, strikes: nc.strikes, event: nc.event }, pitch };
}
