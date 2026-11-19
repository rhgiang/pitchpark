import * as THREE from "three";
import { isOld } from "./old.js";

/*
 * One fan as a low-poly person, and the vertex-shader code that animates it (crowd.js instances it).
 *
 * Local frame: origin at the hip centre, y up, the fan faces -z (toward the field; the seat back is
 * at +z). Feet. Each vertex carries aPart = (part id, arm side): the part picks a per-instance colour
 * (skin / shirt / hair) and the arm side picks which shoulder the vertex swings about.
 * Parts: 0 skin, 1 shirt, 2 hair or cap crown, 3 long hair (only drawn when style = long),
 * 4 cap brim (only when style = cap), 5 eyes, 6 trousers (standing fans' legs).
 *
 * Why colours per part instead of one colour per mesh: a person needs three colours at once, and
 * three separate instanced meshes (head, body, arms, as before) cost three draws per LOD.
 */

export const PART = { SKIN: 0, SHIRT: 1, HAIR: 2, LONG: 3, BRIM: 4, EYE: 5, PANTS: 6 };
/** Hair/cap styles: aAnim.w = STYLE + 2 (the +2 lets the outline pass tell a fan from a seat). */
export const STYLE = { SHORT: 0, LONG: 1, CAP: 2, BALD: 3 };
/** Arm behaviour: rest on the lap, clap now and then, raised (cheering, pumping). */
export const MODE = { REST: 0, CLAP: 1, CHEER: 2 };

export const SHOULDER = { x: 0.6, y: 1.4 }; // pivot of each arm, ft from the hip centre
export const STAND_HIP_FT = 2.5; // hips of a standing fan above the tread (legs fill the gap)

/** Accumulates indexed geometry with smooth normals and per-vertex (part, side). */
class Builder {
  constructor() { this.pos = []; this.nor = []; this.part = []; this.idx = []; }
  vertex(p, n, part, side) {
    this.pos.push(...p); this.nor.push(...n); this.part.push(part, side);
    return this.pos.length / 3 - 1;
  }
  /** Triangle, flipped if needed so it faces the way its vertex normals point. */
  tri(a, b, c) {
    const P = (i) => new THREE.Vector3(this.pos[3 * i], this.pos[3 * i + 1], this.pos[3 * i + 2]);
    const N = (i) => new THREE.Vector3(this.nor[3 * i], this.nor[3 * i + 1], this.nor[3 * i + 2]);
    const f = P(b).sub(P(a)).cross(P(c).sub(P(a)));
    const flip = f.dot(N(a).add(N(b)).add(N(c))) < 0;
    this.idx.push(a, flip ? c : b, flip ? b : c);
  }
  /**
   * A tapered tube along a polyline (smooth radial normals). radii[i] scales ring i; ex/ez are the
   * ring's world-axis scales (a torso is an ellipse in x/z; a brim, along z, is flat in y). capEnd / capStart close the ends.
   */
  tube(path, radii, sides, part, side = 0, { ex = 1, ey = 1, ez = 1, capStart = false, capEnd = false } = {}) {
    const rings = path.map((p, k) => {
      const q = path[Math.min(k + 1, path.length - 1)], o = path[Math.max(k - 1, 0)];
      const t = new THREE.Vector3(q[0] - o[0], q[1] - o[1], q[2] - o[2]).normalize();
      const ref = Math.abs(t.y) < 0.9 ? new THREE.Vector3(0, 1, 0) : new THREE.Vector3(1, 0, 0);
      const u = new THREE.Vector3().crossVectors(t, ref).normalize(), v = new THREE.Vector3().crossVectors(t, u);
      const r = [];
      for (let i = 0; i < sides; i++) {
        const a = (i / sides) * Math.PI * 2, d = u.clone().multiplyScalar(Math.cos(a)).addScaledVector(v, Math.sin(a));
        r.push(this.vertex([p[0] + d.x * ex * radii[k], p[1] + d.y * ey * radii[k], p[2] + d.z * ez * radii[k]],
          new THREE.Vector3(d.x / ex, d.y / ey, d.z / ez).normalize().toArray(), part, side));
      }
      return r;
    });
    for (let k = 0; k + 1 < rings.length; k++) {
      for (let i = 0; i < sides; i++) {
        const j = (i + 1) % sides;
        this.tri(rings[k][i], rings[k][j], rings[k + 1][j]);
        this.tri(rings[k][i], rings[k + 1][j], rings[k + 1][i]);
      }
    }
    const close = (k, dir) => {
      const c = this.vertex(path[k], dir, part, side);
      const ring = rings[k].map((vi) => this.vertex(
        [this.pos[3 * vi], this.pos[3 * vi + 1], this.pos[3 * vi + 2]], dir, part, side));
      for (let i = 0; i < sides; i++) this.tri(c, ring[i], ring[(i + 1) % sides]);
    };
    const dirOf = (a, b) => new THREE.Vector3(b[0] - a[0], b[1] - a[1], b[2] - a[2]).normalize().toArray();
    if (capStart) close(0, dirOf(path[1], path[0]));
    if (capEnd) close(path.length - 1, dirOf(path.at(-2), path.at(-1)));
    return this;
  }
  /** A sphere with smooth normals; partOf(cx, cy, cz) (triangle centroid, unit sphere) picks each face's part. */
  sphere(c, r, ws, hs, partOf, sx = 1, sy = 1) {
    const g = new THREE.SphereGeometry(1, ws, hs), p = g.attributes.position, ix = g.index.array;
    for (let t = 0; t < ix.length; t += 3) {
      const ids = [ix[t], ix[t + 1], ix[t + 2]];
      const cen = ids.reduce((a, i) => [a[0] + p.getX(i) / 3, a[1] + p.getY(i) / 3, a[2] + p.getZ(i) / 3], [0, 0, 0]);
      const part = partOf(...cen);
      const vs = ids.map((i) => {
        const x = p.getX(i), y = p.getY(i), z = p.getZ(i);
        return this.vertex([c[0] + x * r * sx, c[1] + y * r * sy, c[2] + z * r],
          new THREE.Vector3(x / sx, y / sy, z).normalize().toArray(), part, 0);
      });
      if (vs[0] !== vs[1] && vs[1] !== vs[2] && vs[0] !== vs[2]) this.tri(...vs);
    }
    g.dispose();
    return this;
  }
  geometry() {
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.Float32BufferAttribute(this.pos, 3));
    g.setAttribute("normal", new THREE.Float32BufferAttribute(this.nor, 3));
    g.setAttribute("aPart", new THREE.Float32BufferAttribute(this.part, 2));
    g.setIndex(this.idx);
    return g;
  }
}

// Sphere face -> part. Hair covers the crown, the forehead band and the back of the head; the face
// is the lower front. (Unit-sphere centroid coordinates; the fan looks toward -z.)
const headPart = (hairLine) => (x, y, z) => (y > hairLine || z > 0.2 ? PART.HAIR : PART.SKIN);

/**
 * One person, as geometry. detail "near": round head (8x5) with eyes, long-hair and brim parts,
 * 5-sided limbs, arms with sleeves; "far": 6x3 head, 5-sided torso, no arms (crowd.js draws the
 * far arms as their own small mesh, only for fans whose arms move). legs: add trousers (standing).
 */
export function personGeometry(detail, { legs = false } = {}) {
  const b = new Builder(), near = detail === "near";
  b.sphere([0, 2.1, -0.02], 0.4, near ? 8 : 6, near ? 5 : 3, headPart(near ? 0.3 : 0.1), 0.94, 1.04);
  // Torso: an ellipse tube, wider at the shoulders, closed under the head (no neck needed at this size).
  b.tube(near ? [[0, 0, 0], [0, 1.38, 0], [0, 1.72, 0]] : [[0, 0, 0], [0, 1.72, 0]],
    near ? [0.5, 0.6, 0.28] : [0.52, 0.4], near ? 7 : 5, PART.SHIRT, 0, { ex: 1.02, ez: 0.8, capEnd: true });
  if (near) {
    for (const sx of [-1, 1]) { // eyes: two small dark blocks on the face
      b.tube([[sx * 0.14, 2.06, -0.385], [sx * 0.14, 2.06, -0.4]], [0.05, 0.05], 3, PART.EYE, 0, { capEnd: true });
    }
    b.tube([[0, 2.2, 0.27], [0, 1.45, 0.31]], [1, 0.9], 4, PART.LONG, 0, { ex: 0.38, ez: 0.12 });
    b.tube([[0, 2.33, -0.34], [0, 2.3, -0.66]], [1, 0.8], 4, PART.BRIM, 0, { ex: 0.3, ey: 0.03 });
    for (const s of [-1, 1]) arm(b, s, OLD_ARMS ? 5 : 4); // 4 sides: a square-ish toy limb
  }
  if (legs) for (const sx of [-1, 1]) {
    b.tube([[sx * 0.24, 0, -0.05], [sx * 0.26, -STAND_HIP_FT, 0]], [0.24, 0.2], 5, PART.PANTS, 0, { capEnd: true });
  }
  return b.geometry();
}

const OLD_ARMS = isOld("arms");

/** Elbow pivot (right side; x mirrors). The upper arm hangs from SHOULDER to here, beside the torso. */
export const ELBOW = { x: 0.66, y: 0.78, z: -0.06 };

/**
 * One arm (right side s = 1), seated rest pose: the upper arm hangs beside the torso, the elbow bends
 * and the forearm lies forward on the lap, ending in a rounded hand stub (toy-figure style, no fingers).
 * Vertices are tagged side = s on the upper arm and 2s on the forearm, so the shader can bend the
 * elbow (FAN_GLSL): a real arm raised to cheer straightens, and a clap brings the forearms together.
 * Before, the arm was one rigid piece; raised, the lap-bent forearm stuck out sideways like a broken stick.
 * ?old=arms restores that rigid arm.
 */
function arm(b, s, sides, far = false) {
  const S = [s * SHOULDER.x, SHOULDER.y, 0];
  if (OLD_ARMS) {
    const E = [s * 0.72, 0.66, -0.1], H = [s * 0.5, 0.64, -0.92];
    b.tube([S, E], [0.2, 0.17], 4, PART.SHIRT, s);
    b.tube([E, H], [0.16, 0.13], sides, PART.SKIN, s, { capEnd: true });
    return;
  }
  const E = [s * ELBOW.x, ELBOW.y, ELBOW.z], H = [s * 0.5, 0.66, -0.74];
  const lerp = (a, c, t) => a.map((v, i) => v + (c[i] - v) * t);
  if (far) {
    b.tube([S, E], [0.17, 0.15], sides, PART.SHIRT, s);
    b.tube([lerp(E, H, -0.15), H], [0.15, 0.12], sides, PART.SKIN, 2 * s, { capEnd: true });
    return;
  }
  const M = lerp(S, E, 0.45); // short sleeve ends here
  b.tube([lerp(S, E, -0.08), M], [0.18, 0.18], sides, PART.SHIRT, s, { capStart: true });
  b.tube([M, E], [0.15, 0.15], sides, PART.SKIN, s);
  // Forearm starts a little behind the elbow (rounded, so the joint has no gap when it bends).
  b.tube([lerp(E, H, -0.2), lerp(E, H, -0.05), lerp(E, H, 0.85), H], [0.1, 0.15, 0.14, 0.09], sides, PART.SKIN, 2 * s, { capEnd: true });
}

/** Far arms: just the two limbs, fewer sides, for far fans whose arms move. */
export function farArmsGeometry() {
  const b = new Builder();
  for (const s of [-1, 1]) arm(b, s, OLD_ARMS ? 4 : 3, true);
  return b.geometry();
}

/**
 * GLSL for the fan shader: pose (arms swing about the shoulder), bob, and which parts are drawn.
 * Shared by the colour pass (crowd.js) and the outline pre-pass (look.js ndVertexHook), so ink
 * follows the moving arms.
 *
 * Per instance: aAnim = (bob amplitude ft, phase 0..1, arm mode, style + 2).
 * Mode CLAP: bursts of clapping (about a third of the time, per-fan clock); CHEER: arms up, pumping.
 * The bob is the old hop, heavier toward the head (hips 0.5 to head 1.0 of the amplitude).
 */
export const FAN_GLSL = {
  common: /* glsl */`
    uniform float uTime;
    attribute vec4 aAnim;
    attribute vec2 aPart;
    mat3 gR; vec3 gPv; mat3 gRf; vec3 gE; float gBob; float gHide;
    mat3 fxRx(float a) { float c = cos(a), s = sin(a); return mat3(1.0, 0.0, 0.0, 0.0, c, s, 0.0, -s, c); }
    mat3 fxRy(float a) { float c = cos(a), s = sin(a); return mat3(c, 0.0, -s, 0.0, 1.0, 0.0, s, 0.0, c); }
    mat3 fxRz(float a) { float c = cos(a), s = sin(a); return mat3(c, s, 0.0, -s, c, 0.0, 0.0, 0.0, 1.0); }
    void fanPose(vec3 p) {
      gR = mat3(1.0); gPv = vec3(0.0); gRf = mat3(1.0); gE = vec3(0.0); gBob = 0.0; gHide = 0.0;
      if (aAnim.w < 1.5) return; // not a fan (a seat, a tree): attributes absent
      float style = aAnim.w - 2.0, ph = aAnim.y, mode = aAnim.z, part = aPart.x;
      float side = sign(aPart.y), fore = step(1.5, abs(aPart.y)); // forearm vertices carry side 2s
      gHide = (part > 2.5 && part < 3.5 && abs(style - 1.0) > 0.5) || (part > 3.5 && part < 4.5 && abs(style - 2.0) > 0.5) ? 1.0 : 0.0;
      float w = side != 0.0 ? 1.0 : 0.5 + 0.5 * clamp(p.y / 2.2, 0.0, 1.0);
      gBob = aAnim.x * w * pow(0.5 + 0.5 * sin(uTime * (2.6 + ph * 1.4) + ph * 43.0), 2.0);
      if (side != 0.0 && mode > 0.5) {
        // Two rotations: the forearm bends about the elbow (gRf, forearm only), then the whole arm
        // swings about the shoulder (gR). Signs: fxRx(+) lifts a forward-pointing forearm; fxRy(side*+)
        // turns it toward the body's midline; fxRz(side*+) raises a hanging arm out and up.
        gPv = vec3(side * ${SHOULDER.x.toFixed(2)}, ${SHOULDER.y.toFixed(2)}, 0.0);
        gE = vec3(side * ${ELBOW.x.toFixed(2)}, ${ELBOW.y.toFixed(2)}, ${ELBOW.z.toFixed(2)});
        if (mode < 1.5) { // clap: in bursts, upper arms come forward, forearms rise and meet (5 Hz)
          float burst = smoothstep(0.35, 0.5, sin(uTime * 0.27 + ph * 40.0));
          float beat = 0.5 + 0.5 * sin(uTime * (9.0 + ph * 3.0) + ph * 20.0);
          gR = fxRx(burst * 0.35);
          gRf = fxRy(side * burst * (0.45 + 0.4 * beat)) * fxRx(burst * 0.55);
        } else { // cheer: arm straightens and goes up, pumping at the elbow; some fans raise only one
          float one = step(0.7, fract(ph * 7.0)) * step(0.0, side * (fract(ph * 13.0) - 0.5));
          float pump = 0.5 + 0.5 * sin(uTime * (4.0 + ph) + ph * 30.0 + side);
          gR = fxRz(side * (2.55 + 0.25 * pump) * (1.0 - one));
          gRf = fxRx((-1.35 + 0.45 * pump) * (1.0 - one));
        }
        if (fore < 0.5) gRf = mat3(1.0);
      }
    }`,
  normal: /* glsl */`
    fanPose(position);
    objectNormal = gR * gRf * objectNormal;`,
  vertex: /* glsl */`
    transformed = gE + gRf * (transformed - gE);
    transformed = gPv + gR * (transformed - gPv);
    transformed.y += gBob;
    if (gHide > 0.5) transformed = vec3(0.0);`,
};
