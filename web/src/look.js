import * as THREE from "three";
import { isOld } from "./park/old.js";
import { EffectComposer } from "three/examples/jsm/postprocessing/EffectComposer.js";
import { RenderPass } from "three/examples/jsm/postprocessing/RenderPass.js";
import { ShaderPass } from "three/examples/jsm/postprocessing/ShaderPass.js";
import { SMAAPass } from "three/examples/jsm/postprocessing/SMAAPass.js";
import { OutputPass } from "three/examples/jsm/postprocessing/OutputPass.js";
import { UnrealBloomPass } from "three/examples/jsm/postprocessing/UnrealBloomPass.js";
import { SURFACE_KINDS, SURFACE_GLSL } from "./surface.js";

/*
 * The look: one palette, toon materials, outlines, sky and time of day. Everything visible is
 * coloured from PALETTE, so the whole scene stays in one tight set of colours (the thing both
 * references do: Messenger's teal/cream/coral, the cricket park's red/white/green).
 *
 * Pipeline per frame (render()):
 *   1. Normal + depth pass: the outlined objects (layer 0) drawn with one override material that
 *      writes the view-space normal (rgb) and view depth (a) to a float target.
 *   2. RenderPass: the scene as usual (toon materials, fog, sky dome, shadows).
 *   3. Edge pass: ink where the depth, the normal or the object changes abruptly, plus a cheap
 *      ambient occlusion and the colour grade, all from the same buffers (see EDGE_SHADER).
 *   4. Depth of field (intro sweep only; it would blur the ball in the pitch views).
 *   5. Bloom (UnrealBloomPass): only what is brighter than white glows (lamps, lit windows, sun).
 *   6. SMAA (Jimenez et al. 2012): post-process anti-aliasing; MSAA would be lost on the
 *      composer's render targets anyway, and SMAA also smooths the drawn ink lines.
 *   7. OutputPass: linear -> sRGB (no tone mapping: the palette is meant to come out as written).
 */

/** The palette (sRGB hex). Park, pitcher and sky all take their colours from here. */
export const PALETTE = {
  ink: 0x22303a, // outline base colour (a dark blue-grey, as in Messenger, not pure black)
  grass: 0x79b04f, grassMow: 0x6aa346, grassDeep: 0x4f8d45, grassFoul: 0x6ea84c,
  dirt: 0xc9875a, dirtLight: 0xd99c6c, dirtDark: 0xa86d48, chalk: 0xf6f1e3,
  concrete: 0xe7dcc4, concreteShade: 0xc9bca1, paving: 0xd9ccb0,
  wall: 0x2f6b5a, wallTop: 0xf2c14e, batterEye: 0x24493f,
  seat: 0x2f9a96,
  coral: 0xe5604b, cream: 0xf4ead2, navy: 0x2a4a6b, teal: 0x3fa7a0, mustard: 0xf2c14e,
  steel: 0x5d6b72, steelLight: 0x8f9ca1,
  foliage: [0x3f8f4a, 0x2f7a45, 0x5ea24c, 0x7fb24a, 0x2c6b4c], foliageAutumn: [0xe0a33a, 0xd9773a],
  trunk: 0x7a5236, rock: 0x8a6e58, rockDark: 0x5e4a3f, soil: 0x6b4a35,
  water: 0x4fb7c2, waterLight: 0x9fdcd8,
  skin: 0xc98d66, jersey: 0xf3f0e8, pants: 0xe6e1d4, glove: 0x8a5a2b, shoe: 0x1c1c1c,
  crowd: [0xe5604b, 0xf4ead2, 0xf4ead2, 0x3fa7a0, 0xf2c14e, 0xffffff, 0xffffff, 0xd98aa0, 0x6fc3d6, 0x2a4a6b],
  lampOff: 0xdfe6e3, lampOn: 0xfff6d8,
};

// Two-tone toon ramp (+ one in-between for faces lit at a grazing angle). three's toon shader
// looks the ramp up at u = dot(N, L) * 0.5 + 0.5, so the four texels cover N.L in
// [-1,-.5), [-.5,0), [0,.5), [.5,1]: unlit, unlit, grazing, lit. Nearest filtering keeps the
// bands hard, which is what makes it read as flat-shaded rather than smooth.
let ramp = null;
function toonRamp() {
  if (!ramp) {
    ramp = new THREE.DataTexture(new Uint8Array([0, 0, 190, 255]), 4, 1, THREE.RedFormat);
    ramp.minFilter = ramp.magFilter = THREE.NearestFilter;
    ramp.generateMipmaps = false;
    ramp.needsUpdate = true;
  }
  return ramp;
}

// Floodlight falloff, shared by every toon material: at night the sun is replaced by the
// stadium lights, which should light the bowl and not the trees outside it. Direct light is
// scaled from 1 inside radius r0 (ft, around the field centre) to `min` beyond r1. By day min = 1.
// The numbers depend on the park's size, so they come from the park: setParkLighting() below
// (park/index.js calls it). These defaults suit the starter park.
const parkLight = { centreFt: [0, 170], floodRadiiFt: [330, 560], shadowHalfFt: 450 };
const flood = { value: new THREE.Vector4(0, -170, 330, 560) }; // scene x, z, r0, r1
const floodMin = { value: 1 };
// Floodlight pools (night only): each tower lights an elliptical-ish patch, so the field and
// stands at night read as pools of light with dimmer gaps, not one flat wash. Scene x, z, radius.
const MAX_POOLS = 8;
const pools = { value: Array.from({ length: MAX_POOLS }, () => new THREE.Vector3(0, 0, 0)) };
const poolAmt = { value: 0 }; // 0 by day, 1 at night

/**
 * Apply [find, replace] pairs to shader source, throwing if any `find` is missing: a three.js
 * upgrade that renames a chunk should fail loudly, not silently drop the patch.
 */
function patch(src, ...pairs) {
  for (const [a, b] of pairs) {
    if (!src.includes(a)) throw new Error(`look.js shader patch: "${a}" not found`);
    src = src.replace(a, b);
  }
  return src;
}

// Painted surface texture (plaster, foliage, wood, ground, rock): see surface.js.
// Direction toward the key light (world), for the ground's bump (surface.js groundSurface).
const sunDir = { value: new THREE.Vector3(0, 1, 0) };

function addParkShading(shader) {
  shader.uniforms.uSunDir = sunDir;
  shader.uniforms.uFlood = flood;
  shader.uniforms.uFloodMin = floodMin;
  shader.uniforms.uPools = pools;
  shader.uniforms.uPoolAmt = poolAmt;
  shader.vertexShader = patch(shader.vertexShader,
    ["#include <common>", "#include <common>\nvarying vec3 vParkW;\n#if SURFACE == 6\nvarying vec3 vParkN;\n#endif"],
    ["#include <project_vertex>", `#include <project_vertex>
      vec4 parkW = vec4(transformed, 1.0);
      #ifdef USE_INSTANCING
        parkW = instanceMatrix * parkW;
      #endif
      vParkW = (modelMatrix * parkW).xyz;
      #if SURFACE == 6
        // World normal, for the brick's horizontal direction (surface.js): faces here are flat, so this is constant across one.
        vec3 parkN = objectNormal;
        #ifdef USE_INSTANCING
          parkN = mat3(instanceMatrix) * parkN;
        #endif
        vParkN = normalize(mat3(modelMatrix) * parkN);
      #endif`]);
  shader.fragmentShader = patch(shader.fragmentShader,
    ["#include <common>", `#include <common>
      varying vec3 vParkW;
      #if SURFACE == 6
        varying vec3 vParkN;
      #endif
      uniform vec4 uFlood;
      uniform float uFloodMin, uPoolAmt;
      uniform vec3 uPools[${MAX_POOLS}];
      uniform vec3 uSunDir;
      ${SURFACE_GLSL}`],
    ["#include <color_fragment>", `#include <color_fragment>
      #if SURFACE == 4
        diffuseColor.rgb = groundSurface(diffuseColor.rgb, vParkW);
      #elif SURFACE == 6
        diffuseColor.rgb = brickSurface(diffuseColor.rgb, vParkW, normalize(vParkN));
      #else
        diffuseColor.rgb *= surfaceTone(vParkW);
      #endif`],
    ["#include <lights_fragment_end>", `#include <lights_fragment_end>
      #if SURFACE == 4
        reflectedLight.directDiffuse *= groundLight; // the ground's small relief (surface.js)
      #endif
      reflectedLight.directDiffuse *= mix(1.0, uFloodMin,
        smoothstep(uFlood.z, uFlood.w, distance(vParkW.xz, uFlood.xy)));
      float pool = 0.0;
      for (int i = 0; i < ${MAX_POOLS}; i++)
        pool = max(pool, 1.0 - smoothstep(0.35 * uPools[i].z, uPools[i].z, distance(vParkW.xz, uPools[i].xy)));
      reflectedLight.directDiffuse *= mix(1.0, 0.45 + 0.85 * pool, uPoolAmt);`]);
}

/**
 * Lighting numbers that depend on the park, in Statcast feet; any omitted keep their value.
 * centreFt: middle of the bowl (night falloff centre, sun aim); floodRadiiFt: [r0, r1], full
 * floodlight inside r0, faded out by r1; shadowHalfFt: half-width of the sun's shadow box;
 * poolsFt: up to 8 night floodlight pools, [x, y, radius] each (where the towers aim).
 * Works before or after initLook.
 */
export function setParkLighting(opts = {}) {
  for (const k of ["centreFt", "floodRadiiFt", "shadowHalfFt"]) if (opts[k] !== undefined) parkLight[k] = opts[k];
  const [x, y] = parkLight.centreFt;
  flood.value.set(x, -y, ...(look.flood ?? parkLight.floodRadiiFt));
  if (opts.poolsFt) {
    pools.value.forEach((v, i) => {
      const p = opts.poolsFt[i];
      v.set(p ? p[0] : 0, p ? -p[1] : 0, p ? p[2] : 0); // radius 0: no pool
    });
  }
  if (!look.sun) return; // initLook reads parkLight when it runs
  look.sun.target.position.set(x, 0, -y);
  const h = parkLight.shadowHalfFt;
  Object.assign(look.sun.shadow.camera, { left: -h, right: h, top: h, bottom: -h });
  look.sun.shadow.camera.updateProjectionMatrix();
  if (look.timeOfDay) setTimeOfDay(look.timeOfDay); // re-place the sun about the new centre
}

/**
 * A toon material in the park's style: the shared two-tone ramp, painted surface texture, fog,
 * the night falloff and floodlight pools.
 * @param color hex or THREE.Color (use PALETTE)
 * @param extra other MeshToonMaterial parameters (map, vertexColors, side, ...), plus
 *   `surface`: "plaster" (default), "foliage", "wood", "ground", "rock", "brick" or "none" (surface.js).
 */
export function makeToonMaterial(color, { surface = "plaster", ...extra } = {}) {
  const m = new THREE.MeshToonMaterial({ color, gradientMap: toonRamp(), ...extra });
  m.defines = { SURFACE: SURFACE_KINDS[surface] ?? 1 };
  m.onBeforeCompile = addParkShading;
  return m;
}

// Windows: dark by day; at night most glow warm (lit rooms and concourses), the rest stay dark.
export const windowMaterial = makeToonMaterial(PALETTE.navy, { surface: "none" });
export const windowLitMaterial = makeToonMaterial(PALETTE.navy, { surface: "none" });
// setTimeOfDay recolours these two, so a mesh merge (park/build.js) must keep them by identity.
windowMaterial.userData.shared = windowLitMaterial.userData.shared = true;

// Lamps on the light towers: pale by day, glowing at night (setTimeOfDay switches them).
export const lampMaterial = new THREE.MeshBasicMaterial({ color: PALETTE.lampOff });
export const glowMaterial = new THREE.SpriteMaterial({
  color: 0xfff1c8, transparent: true, blending: THREE.AdditiveBlending, depthWrite: false,
  opacity: 0, fog: false,
});

const glowSprites = []; // setTimeOfDay rescales them (the night halo is bigger)

/** A soft additive glow (for lamps), visible at sunset and night only. Browser only (canvas). */
export function makeGlowSprite(sizeFt) {
  if (!glowMaterial.map) {
    const c = document.createElement("canvas");
    c.width = c.height = 128;
    const g = c.getContext("2d");
    const r = g.createRadialGradient(64, 64, 0, 64, 64, 64);
    r.addColorStop(0, "rgba(255,255,255,1)");
    r.addColorStop(0.25, "rgba(255,255,255,0.45)");
    r.addColorStop(1, "rgba(255,255,255,0)");
    g.fillStyle = r;
    g.fillRect(0, 0, 128, 128);
    glowMaterial.map = new THREE.CanvasTexture(c);
  }
  const s = new THREE.Sprite(glowMaterial);
  s.scale.set(sizeFt, sizeFt, 1);
  s.userData.baseFt = sizeFt;
  glowSprites.push(s);
  return noOutline(s);
}

/**
 * Layer 2: never outlined and never inked over (ball, ghost, strike zone). These are what the
 * viewer is about, so they must stay on top of every line: the normal/depth pass marks their
 * pixels (a < 0) wherever they are nearer than the outlined scene, and the edge pass skips ink
 * there and treats marked neighbours as background-free (no ring around the ball either).
 */
export function inkProof(obj) {
  obj.traverse((o) => o.layers.set(2));
  return obj;
}

/*
 * Terrain: the ground and what is shaped from it (painted ground, mound, plate, rubber, bullpen
 * mounds). A real mound has no line round it; ground features should meet through colour,
 * texture and contact shading, not ink (Ryan, 2026-11-13). Terrain writes TERRAIN_ID as its
 * object ID in the outline pre-pass, and the edge pass draws no ink on terrain pixels except
 * against the sky (the island's rim). Objects standing on terrain (pitcher, bases, walls) are
 * still inked, on their own side of the line.
 */
export const TERRAIN_ID = -1;

/** Mark every mesh under obj as terrain (a per-vertex objId = TERRAIN_ID, read by the pre-pass). */
export function terrain(obj) {
  obj.traverse((o) => {
    if (!o.isMesh) return;
    o.userData.terrain = true;
    const n = o.geometry.attributes.position.count;
    o.geometry.setAttribute("objId", new THREE.Float32BufferAttribute(new Float32Array(n).fill(TERRAIN_ID), 1));
  });
  return obj;
}

/** Whether mesh was marked by terrain(). */
export const isTerrain = (mesh) => mesh.userData.terrain === true;

/** Layer 1 only: drawn, but not outlined (sky, glows, the strike zone, the ball, nets). */
export function noOutline(obj) {
  obj.traverse((o) => o.layers.set(1));
  return obj;
}

// ---------- outlines ----------

// Normal, object ID and depth for the edge pass: rg = view-space normal x, y (z is recovered as
// +sqrt(1 - x^2 - y^2): DoubleSide flips every visible face toward the camera), b = an object ID,
// a = view depth / far. The ID is a hash of the object's model matrix and, for instanced meshes,
// the instance index (no per-object uniforms needed), so a window on a wall, a sign on a fascia or
// one tree in front of another gets a line even where the depth step is too small to see.
// transparent + NoBlending: keeps three from forcing alpha to 1 (it does for opaque materials).
// The crowd's animated fans (park/crowd.js sets this): the pre-pass runs their vertex animation too, so ink
// follows the swinging arms. Applies only to instanced meshes with an instanceColor and a fan aAnim.
export const fanHook = { glsl: null, uTime: null, lite: false };

function makeNormalDepthMaterial(far) {
  const m = new THREE.MeshNormalMaterial({ side: THREE.DoubleSide, transparent: true, blending: THREE.NoBlending });
  m.onBeforeCompile = (s) => {
    s.uniforms.uFar = { value: far };
    if (fanHook.glsl) {
      s.uniforms.uTime = fanHook.uTime;
      s.vertexShader = patch(s.vertexShader,
        ["#include <common>", `#include <common>
${fanHook.glsl.common}`],
        ["#include <beginnormal_vertex>", `#include <beginnormal_vertex>
#ifdef USE_INSTANCING_COLOR
${fanHook.glsl.normal}
#endif`],
        ["#include <begin_vertex>", `#include <begin_vertex>
#ifdef USE_INSTANCING_COLOR
${fanHook.glsl.vertex}
#endif`]);
    }
    s.vertexShader = patch(s.vertexShader,
      ["#include <common>", "#include <common>\nvarying float vViewZ;\nflat varying float vId;\n// Merged park meshes carry one ID per source object (park/build.js mergeStatic); 0 elsewhere.\nattribute float objId;"],
      ["#include <project_vertex>", `#include <project_vertex>
        vViewZ = -mvPosition.z;
        float inst = 0.0;
        #ifdef USE_INSTANCING
          inst = float(gl_InstanceID);
        #endif
        vId = fract(sin(dot(modelMatrix[3].xyz, vec3(12.9898, 78.233, 37.719))
          + dot(modelMatrix[0].xyz, vec3(3.17, 5.71, 7.13)) + inst * 0.6180339 + objId * 0.7548776) * 43758.5453);
        if (objId < -0.5) vId = -1.0; // terrain (TERRAIN_ID): one shared, negative ID`]);
    s.fragmentShader = patch(s.fragmentShader,
      ["#include <packing>", "#include <packing>\nvarying float vViewZ;\nflat varying float vId;\nuniform float uFar;"],
      ["gl_FragColor = vec4( packNormalToRGB( normal ), diffuseColor.a );",
        "gl_FragColor = vec4( normal.xy, vId, vViewZ / uFar );"]);
  };
  return m;
}

/*
 * Edge detection, in screen space. Why not an inverted hull (a back-face-scaled copy of every
 * mesh)? It doubles every draw call, needs per-mesh thickness tuning, misses creases (the step
 * edges of the stands, box corners), and would have to be rebuilt for instanced crowds. One
 * extra pass on normals + depth outlines everything, at a constant pixel width.
 *
 * Depth edges use the second difference of 1/z. For any plane, 1/z (view depth) is an affine
 * function of screen position, so its second difference is zero on every plane however steeply
 * it is seen: the grass at a grazing angle gives no false edges, but a silhouette (a jump in z)
 * or a convex crease (a ridge in 1/z) does. Only the negative side counts (the pixel is nearer
 * than its neighbours' average), so the line sits on the object's own side of its silhouette,
 * as an inked outline does, not on whatever is behind it. Testing at 1 and 2 px gives a line
 * about 2 px wide (x uScale). Normal edges catch concave creases (wall meets floor). Lines fade
 * with distance so far-away crowds and seat rows don't turn to grey mush.
 */
const EDGE_SHADER = {
  uniforms: {
    tDiffuse: { value: null },
    tND: { value: null },
    uTexel: { value: new THREE.Vector2() },
    uScale: { value: 1 }, // line width multiplier (1 at 900 px tall)
    uInk: { value: new THREE.Color(PALETTE.ink) },
    uFar: { value: 1 },
    uFade: { value: new THREE.Vector2(260, 2200) }, // ft: 2 px lines until 260, then 1 px, 40% by 2200
    uAO: { value: 0.35 }, // how dark the screen-space occlusion gets in creases and at contacts
    uShadowTint: { value: new THREE.Color(1, 1, 1) }, uLightTint: { value: new THREE.Color(1, 1, 1) },
    uSat: { value: 1 }, uVignette: { value: 0.2 },
  },
  vertexShader: /* glsl */`
    varying vec2 vUv;
    void main() { vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`,
  fragmentShader: /* glsl */`
    uniform sampler2D tDiffuse, tND;
    uniform vec2 uTexel, uFade;
    uniform vec3 uInk;
    uniform float uFar, uScale, uAO, uSat, uVignette;
    uniform vec3 uShadowTint, uLightTint;
    varying vec2 vUv;
    vec4 c0; // the centre sample; an ink-proof neighbour (a < 0) reads as the centre
    vec4 nd(vec2 o) { vec4 v = textureLod(tND, vUv + o * uTexel * uScale, 0.0); return v.a < 0.0 ? c0 : v; }
    float inv(float a) { return 1.0 / max(a, 1e-6); }
    // Near-side silhouette / ridge strength at step k px along both axes.
    float ridge(float iz, float k) {
      float sx = inv(nd(vec2(-k, 0.0)).a) + inv(nd(vec2(k, 0.0)).a) - 2.0 * iz;
      float sy = inv(nd(vec2(0.0, -k)).a) + inv(nd(vec2(0.0, k)).a) - 2.0 * iz;
      return max(-sx, -sy) / iz;
    }
    vec3 nrm(vec4 v) { return vec3(v.xy, sqrt(max(0.0, 1.0 - dot(v.xy, v.xy)))); }
    // Another object in front of this pixel's object, or level with it, at step k px: ink on the
    // nearer side only (as with the depth ridge), so a window on a wall gets its line, not the wall.
    float idOne(vec2 o) {
      vec4 n = nd(o);
      return n.a < 1.0 && abs(n.b - c0.b) > 1e-3 && c0.a <= n.a * 1.0005 ? 1.0 : 0.0;
    }
    float idEdge(float k) {
      return max(max(idOne(vec2(k, 0.0)), idOne(vec2(-k, 0.0))), max(idOne(vec2(0.0, k)), idOne(vec2(0.0, -k))));
    }
    /*
     * Cheap ambient occlusion from the same normal/depth buffer (no extra scene pass): in four
     * directions, a pixel whose neighbours ~3 ft away are on average nearer than it, in 1/z, sits in
     * a concave spot (a crease, the foot of a wall, under a seat) and is darkened. 1/z is affine on
     * any plane, so flat ground at a grazing angle stays clean. Neighbours more than ~10 ft nearer
     * are a different object in front, not a crease, and are ignored (no dark halos).
     */
    float occlusion(float iz, float distFt) {
      float rpx = clamp(3000.0 / distFt, 2.0, 26.0);
      float occ = 0.0;
      for (int i = 0; i < 4; i++) {
        float a = float(i) * 0.785398;
        vec2 d = vec2(cos(a), sin(a)) * rpx;
        // Raw samples: skip sky and ink-proof pixels (ball, zone) rather than substituting.
        vec4 p = textureLod(tND, vUv + d * uTexel * uScale, 0.0), m = textureLod(tND, vUv - d * uTexel * uScale, 0.0);
        if (p.a >= 1.0 || m.a >= 1.0 || p.a < 0.0 || m.a < 0.0) continue;
        float conc = (inv(p.a) + inv(m.a) - 2.0 * iz) / iz;
        float jump = max(c0.a - p.a, c0.a - m.a) * uFar;
        occ += smoothstep(0.004, 0.05, conc) * (1.0 - smoothstep(6.0, 14.0, jump));
      }
      return occ * 0.25;
    }
    /** Warm, consistent grade: a little saturation, cool shadows, warm lights, a soft vignette. */
    vec3 grade(vec3 col) {
      float lum = dot(col, vec3(0.2126, 0.7152, 0.0722));
      col = mix(vec3(lum), col, uSat);
      col *= mix(uShadowTint, uLightTint, smoothstep(0.03, 0.5, lum));
      return col * (1.0 - uVignette * smoothstep(0.35, 0.95, length((vUv - 0.5) * vec2(1.3, 1.0))));
    }
    void main() {
      vec3 col = texture2D(tDiffuse, vUv).rgb;
      c0 = texture2D(tND, vUv);
      if (c0.a < 0.0) { gl_FragColor = vec4(col, 1.0); return; } // ball / ghost / zone: no ink, no grade
      vec4 c = c0;
      if (c.a >= 1.0) { gl_FragColor = vec4(grade(col), 1.0); return; } // sky: no line
      float iz = inv(c.a);
      float distFt = c.a * uFar;
      col *= 1.0 - uAO * occlusion(iz, distFt) * (1.0 - smoothstep(400.0, 1200.0, distFt));
      // Line width: 2 px near, 1 px far (fewer samples), so far crowds and seats stay legible.
      float far = step(uFade.x, distFt);
      float depthEdge = smoothstep(0.02, 0.06, max(ridge(iz, 1.0), ridge(iz, 2.0 - far)));
      // Object edges only near: at range every seat and spectator is its own object, and inking
      // each one turns the stands to mush (depth edges still outline the bowl there).
      float objectEdge = max(idEdge(1.0), idEdge(2.0 - far)) * (1.0 - smoothstep(140.0, 320.0, distFt));
      vec4 l = nd(vec2(-1.0, 0.0)), r = nd(vec2(1.0, 0.0)), d = nd(vec2(0.0, -1.0)), u = nd(vec2(0.0, 1.0));
      // Normals only compared with neighbours on the same surface patch (not across the sky).
      float nDiff = 0.0;
      vec3 cn = nrm(c);
      if (max(max(l.a, r.a), max(d.a, u.a)) < 1.0)
        nDiff = max(max(1.0 - dot(cn, nrm(l)), 1.0 - dot(cn, nrm(r))),
                    max(1.0 - dot(cn, nrm(d)), 1.0 - dot(cn, nrm(u))));
      float normalEdge = smoothstep(0.15, 0.35, nDiff);
      // Lines stay to the far side of the island, thinning past uFade.x and fading by uFade.y.
      // Varied weight, as with a brush: silhouettes and object outlines full strength, creases
      // inside one object (normal edges) lighter.
      float edge = max(max(depthEdge, objectEdge), 0.7 * normalEdge);
      // Terrain (b = TERRAIN_ID): no ink where ground meets ground (the mound, the plate, the
      // painted layers); only its silhouette against the sky, so the island keeps its rim.
      if (c.b < 0.0) edge = step(1.0, max(max(l.a, r.a), max(d.a, u.a)));
      edge *= 1.0 - 0.6 * smoothstep(uFade.x, uFade.y, distFt);
      // Ink takes a little of the colour under it (as hand-inked lines do), so lines on dark
      // surfaces stay dark and lines on cream read as warm brown-grey.
      vec3 ink = mix(uInk, col * 0.3, 0.2);
      gl_FragColor = vec4(grade(mix(col, ink, edge)), 1.0);
    }`,
};

/*
 * Depth of field, for the intro sweep only (never in the pitch views: it would blur the ball). A
 * disc blur whose radius grows with distance from the focus plane, read from the same depth buffer
 * as the ink, so the wide shot looks like a tilt-shift miniature (as the cricket clip's does).
 * 16 taps on a golden-angle spiral (Vogel disc).
 */
const DOF_SHADER = {
  uniforms: {
    tDiffuse: { value: null }, tND: { value: null }, uTexel: { value: new THREE.Vector2() },
    uFar: { value: 1 }, uFocus: { value: 100 }, uMaxPx: { value: 0 },
  },
  vertexShader: EDGE_SHADER.vertexShader,
  fragmentShader: /* glsl */`
    uniform sampler2D tDiffuse, tND;
    uniform vec2 uTexel;
    uniform float uFar, uFocus, uMaxPx;
    varying vec2 vUv;
    float blurPx(vec2 uv) {
      float a = textureLod(tND, uv, 0.0).a;
      float z = (a < 0.0 ? uFocus / uFar : a) * uFar; // ink-proof pixels: in focus
      return uMaxPx * clamp(abs(z - uFocus) / (uFocus * 0.6), 0.0, 1.0);
    }
    void main() {
      float r = blurPx(vUv);
      vec3 sum = texture2D(tDiffuse, vUv).rgb;
      float w = 1.0;
      for (int i = 1; i < 16; i++) {
        float k = float(i);
        vec2 o = vec2(cos(k * 2.39996), sin(k * 2.39996)) * sqrt(k / 16.0) * r * uTexel;
        sum += textureLod(tDiffuse, vUv + o, 0.0).rgb;
        w += 1.0;
      }
      gl_FragColor = vec4(sum / w, 1.0);
    }`,
};

// ---------- sky ----------

/*
 * Sky dome: a banded gradient (posterised into flat steps, like the cricket clip's sunset), a
 * flat sun or moon disc with one halo ring, two-tone toon clouds from value noise (lit tops,
 * shaded undersides, Messenger-style streaks stretched along x), and stars at night.
 */
const SKY_SHADER = {
  uniforms: {
    uTop: { value: new THREE.Color() }, uMid: { value: new THREE.Color() },
    uHorizon: { value: new THREE.Color() }, uBelow: { value: new THREE.Color() },
    uSunDir: { value: new THREE.Vector3(0, 1, 0) }, uSunColor: { value: new THREE.Color() },
    uSunSize: { value: 0.03 },
    uCloud: { value: new THREE.Color() }, uCloudShade: { value: new THREE.Color() },
    uCloudCover: { value: 0.55 }, uStars: { value: 0 }, uStarCut: { value: 0.985 },
  },
  vertexShader: /* glsl */`
    varying vec3 vDir;
    void main() {
      vDir = position;
      vec4 p = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      gl_Position = p.xyww; // on the far plane
    }`,
  fragmentShader: /* glsl */`
    uniform vec3 uTop, uMid, uHorizon, uBelow, uSunDir, uSunColor, uCloud, uCloudShade;
    uniform float uSunSize, uCloudCover, uStars, uStarCut;
    varying vec3 vDir;
    float hash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
    float noise(vec2 p) {
      vec2 i = floor(p), f = fract(p);
      f = f * f * (3.0 - 2.0 * f);
      return mix(mix(hash(i), hash(i + vec2(1, 0)), f.x), mix(hash(i + vec2(0, 1)), hash(i + vec2(1, 1)), f.x), f.y);
    }
    float fbm(vec2 p) {
      float v = 0.0, a = 0.5;
      for (int i = 0; i < 5; i++) { v += a * noise(p); p = p * 2.03 + 17.1; a *= 0.5; }
      return v;
    }
    void main() {
      vec3 d = normalize(vDir);
      float h = d.y;
      // Posterised gradient: 7 flat bands between horizon and zenith.
      float t = floor(clamp(h, 0.0, 1.0) * 7.0) / 7.0;
      vec3 col = h >= 0.0
        ? mix(mix(uHorizon, uMid, smoothstep(0.0, 0.3, t)), uTop, smoothstep(0.3, 0.9, t))
        : mix(uHorizon, uBelow, smoothstep(0.0, 0.35, floor(-h * 6.0) / 6.0));
      // Sun / moon: a flat disc and one flat halo ring.
      float s = dot(d, normalize(uSunDir));
      col = mix(col, mix(col, uSunColor, 0.35), step(1.0 - uSunSize * 4.0, s));
      col = mix(col, uSunColor * 1.8, step(1.0 - uSunSize, s)); // > 1: blooms
      // Clouds on a plane above (and, thinner, below) the island.
      float ah = abs(h);
      vec2 p = d.xz / (ah + 0.12) * vec2(0.55, 1.6);
      float n = fbm(p * 1.3);
      float cover = uCloudCover + (h < 0.0 ? 0.2 : 0.0);
      float cloud = step(cover, n) * smoothstep(0.02, 0.12, ah);
      float shade = step(cover + 0.03, fbm(p * 1.3 + vec2(0.0, 0.06)));
      vec3 cc = mix(uCloudShade, uCloud, shade);
      col = mix(col, cc, cloud);
      // Stars: one hash per grid cell of direction, a few per cent lit.
      vec2 sp = vec2(atan(d.z, d.x), asin(clamp(h, -1.0, 1.0))) * 300.0;
      vec2 sg = floor(sp);
      float hs = hash(sg);
      float star = step(uStarCut, hs) * step(length(fract(sp) - 0.5), 0.22) // a round dot per lit cell
        * smoothstep(0.05, 0.3, h) * (1.0 - cloud);
      // New night: stars warmer or bluer (the hash's fraction picks), a few brighter. ?old=night: flat white.
      bool newSky = uStarCut < 0.98;
      vec3 starCol = newSky ? mix(vec3(0.75, 0.85, 1.0), vec3(1.0, 0.92, 0.75), fract(hs * 97.0)) : vec3(1.0);
      col += uStars * star * starCol * (newSky ? (hs > 0.995 ? 1.1 : 0.7) : 0.9);
      gl_FragColor = vec4(col, 1.0);
      #include <colorspace_fragment>
    }`,
};

// ---------- time of day ----------

/*
 * Each time of day is a full recolour: sky bands, clouds, fog, sun (direction, colour,
 * intensity), ambient (hemisphere) and outline ink. Intensities: three's lights are physically
 * scaled, so a white directional light of intensity pi on a face pointing at it reproduces the
 * material colour exactly; lit = (sun + ambient) / pi of the palette colour.
 * Sun direction: [azimuth deg from +x toward +z (scene), elevation deg].
 */
export const TIMES = {
  day: {
    top: 0x2a9fb8, mid: 0x5cc5c3, horizon: 0xb4e6da, below: 0xa9dcd6,
    cloud: 0xf5fbf4, cloudShade: 0xbfe4df, cover: 0.56, stars: 0,
    sun: [150, 55], sunColor: 0xfff4e0, sunIntensity: 2.0, sunSize: 0.0012,
    sky: 0xd8f0ff, ground: 0x8a9a6a, ambient: 1.35,
    fog: 0xb4e6da, fogDensity: 0.00022, ink: 0x22303a, floodMin: 1, lamps: false,
    // Grade: shadows pulled cool, lights warm (the courtyard reference's sunlit warmth).
    shadowTint: [0.93, 0.97, 1.06], lightTint: [1.06, 1.02, 0.94], sat: 1.06, bloom: 0.35,
  },
  sunset: {
    top: 0x4a3b80, mid: 0xc8708f, horizon: 0xffbe7d, below: 0xf2a27e,
    cloud: 0xffc6a6, cloudShade: 0xa9628f, cover: 0.5, stars: 0,
    // Setting over left-centre field: low, behind the pitcher's glove side, so the bowl faces
    // into it and the left-field stands and light towers throw long shadows across the
    // outfield toward the infield (a sun behind home would put the whole infield in shadow).
    sun: [-128, 9], sunColor: 0xffb070, sunIntensity: 2.6, sunSize: 0.0016,
    sky: 0xf0b4c0, ground: 0xb08070, ambient: 1.5,
    fog: 0xf0a47e, fogDensity: 0.0003, ink: 0x3a2433, floodMin: 1, lamps: true,
    shadowTint: [0.92, 0.9, 1.08], lightTint: [1.08, 1.0, 0.9], sat: 1.05, bloom: 0.6,
  },
  // Night game: deep blue sky with stars, the field warm and bright under the towers, the
  // bowl and everything outside it falling away into blue-black (flood: fully lit out to 200 ft
  // of the field centre, gone by 380, so the front rows catch the light and the upper decks do not).
  night: {
    top: 0x03081a, mid: 0x0b1840, horizon: 0x2a4278, below: 0x0c1430,
    cloud: 0x1a2a55, cloudShade: 0x0f1a38, cover: 0.66, stars: 1, starCut: 0.972,
    sun: [-90, 62], sunColor: 0xfff0d6, sunIntensity: 2.3, sunSize: 0.0009, // key: warm floodlight, from centre field, high
    sky: 0x3b52a0, ground: 0x0e1426, ambient: 0.3,
    fog: 0x0f1a3a, fogDensity: 0.00034, ink: 0x03050b, floodMin: 0.03, lamps: true,
    flood: [200, 380], lampGain: 6, glowOpacity: 0.55, glowScale: 2.2,
    fill: [90, 45], fillIntensity: 1.2, // floodlights from behind home too, so faces toward the plate aren't black
    shadowTint: [0.85, 0.94, 1.14], lightTint: [1.06, 1.02, 0.92], sat: 1.06, bloom: 0.85, pools: true,
  },
};
/** The night as it was before the 2026-11 scene pass: ?old=night restores it (A/B comparison). */
const OLD_NIGHT = {
    top: 0x070d22, mid: 0x14204a, horizon: 0x2b3f6e, below: 0x1a2648,
    cloud: 0x2c3c66, cloudShade: 0x1b264a, cover: 0.62, stars: 1,
    sun: [-90, 62], sunColor: 0xf0f4ff, sunIntensity: 1.9, sunSize: 0.0009, // key: from centre field, high
    sky: 0x5a6ea8, ground: 0x1c2438, ambient: 0.5,
    fog: 0x1f2d55, fogDensity: 0.0003, ink: 0x05070d, floodMin: 0.05, lamps: true,
    fill: [90, 45], fillIntensity: 1.1, // floodlights from behind home too, so faces toward the plate aren't black
    shadowTint: [0.88, 0.95, 1.12], lightTint: [1.06, 1.02, 0.94], sat: 1.08, bloom: 0.75, pools: true,
  
};
// At night the "sun" uniform draws the moon; the key light is the floodlights, from high above
// the field (the direction in `sun` above), so the moon is placed separately.
const MOON = [-60, 35];

const look = {};

const dirFrom = ([az, el]) => {
  const a = (az * Math.PI) / 180, e = (el * Math.PI) / 180;
  return new THREE.Vector3(Math.cos(e) * Math.cos(a), Math.sin(e), Math.cos(e) * Math.sin(a));
};

/**
 * Set the time of day: "day" | "sunset" | "night". Recolours sky, clouds, fog, sun, ambient
 * and ink, and switches the light towers on at sunset and night. Call after initLook.
 */
export function setTimeOfDay(name) {
  if (!Object.hasOwn(TIMES, name)) name = "day"; // ?tod= is user input: own keys only
  const T = name === "night" && isOld("night") ? OLD_NIGHT : TIMES[name];
  const u = look.sky.material.uniforms;
  u.uTop.value.set(T.top); u.uMid.value.set(T.mid);
  u.uHorizon.value.set(T.horizon); u.uBelow.value.set(T.below);
  u.uCloud.value.set(T.cloud); u.uCloudShade.value.set(T.cloudShade);
  u.uCloudCover.value = T.cover; u.uStars.value = T.stars; u.uStarCut.value = T.starCut ?? 0.985;
  u.uSunDir.value.copy(dirFrom(name === "night" ? MOON : T.sun));
  u.uSunColor.value.set(name === "night" ? 0xeef2ff : T.sunColor);
  u.uSunSize.value = T.sunSize;
  sunDir.value.copy(dirFrom(T.sun));
  look.sun.position.copy(dirFrom(T.sun).multiplyScalar(1500)).add(look.sun.target.position);
  look.sun.color.set(T.sunColor);
  look.sun.intensity = T.sunIntensity;
  look.fill.intensity = T.fillIntensity ?? 0;
  if (T.fill) look.fill.position.copy(dirFrom(T.fill).multiplyScalar(1500)).add(look.sun.target.position);
  look.fill.color.set(T.sunColor);
  look.hemi.color.set(T.sky); look.hemi.groundColor.set(T.ground);
  look.hemi.intensity = T.ambient;
  look.scene.fog.color.set(T.fog); look.scene.fog.density = T.fogDensity;
  look.renderer.setClearColor(T.fog, 1);
  look.edge.uniforms.uInk.value.set(T.ink);
  floodMin.value = T.floodMin;
  // Where the floodlights fade out: the park's radii, or this time of day's own (night draws it in).
  flood.value.z = (T.flood ?? parkLight.floodRadiiFt)[0]; flood.value.w = (T.flood ?? parkLight.floodRadiiFt)[1];
  look.flood = T.flood;
  // Lit lamps and windows are brighter than white, which is what the bloom pass picks up.
  lampMaterial.color.set(T.lamps ? PALETTE.lampOn : PALETTE.lampOff).multiplyScalar(T.lamps ? (T.lampGain ?? 3) : 1);
  windowLitMaterial.emissive.set(name === "night" ? 0xffb85c : T.lamps ? 0x8a5a30 : 0x000000);
  windowLitMaterial.emissiveIntensity = name === "night" ? 1.6 : 1;
  poolAmt.value = T.pools ? 1 : 0;
  const e = look.edge.uniforms;
  e.uShadowTint.value.setRGB(...T.shadowTint); e.uLightTint.value.setRGB(...T.lightTint);
  e.uSat.value = T.sat;
  look.bloom.strength = T.bloom;
  glowMaterial.opacity = T.glowOpacity ?? (name === "night" ? 0.15 : T.lamps ? 0.12 : 0); // bloom does most of the glow
  for (const sp of glowSprites) sp.scale.setScalar(sp.userData.baseFt * (T.glowScale ?? 1));
  look.timeOfDay = name;
  refreshShadows();
}

/** Redraw the shadow map on the next frame (it is only redrawn on demand: the park is static). */
export function refreshShadows() {
  if (look.renderer) look.renderer.shadowMap.needsUpdate = true;
}

/** The current time of day's name. */
export const timeOfDay = () => look.timeOfDay;

/**
 * Set up lights, fog, sky and the post-processing chain for `scene` seen by `camera`.
 * @returns {{render: () => void, setSize: (w: number, h: number) => void,
 *   setFocus: (focusFt: number | null, maxPx?: number) => void, setLite: (on: boolean) => void}}
 *   setFocus: depth of field (intro only); setLite: the lighter-graphics mode
 */
export function initLook(renderer, scene, camera) {
  Object.assign(look, { renderer, scene, camera });
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFShadowMap; // crisp-ish edges suit the toon look
  renderer.shadowMap.autoUpdate = false; // static park: redraw on pose / time-of-day change only
  renderer.toneMapping = THREE.NoToneMapping;

  scene.fog = new THREE.FogExp2(0xffffff, 0.0002);
  look.hemi = new THREE.HemisphereLight(0xffffff, 0x000000, 1);
  scene.add(look.hemi);
  // One shadow-casting sun over the stadium (900 ft square): 4096 texels -> ~0.22 ft per texel,
  // enough for the pitcher's shadow on the mound. Trees beyond have painted contact shadows.
  const sun = new THREE.DirectionalLight(0xffffff, 2);
  sun.target.position.set(parkLight.centreFt[0], 0, -parkLight.centreFt[1]);
  sun.castShadow = true;
  sun.shadow.mapSize.set(4096, 4096);
  const h = parkLight.shadowHalfFt;
  Object.assign(sun.shadow.camera, { left: -h, right: h, top: h, bottom: -h, near: 900, far: 2100 });
  // Biases against shadow acne on walls the sun grazes (~0.3 ft texels): a constant depth
  // offset plus a push along the surface normal, in feet.
  sun.shadow.bias = -0.0005;
  sun.shadow.normalBias = 0.9;
  scene.add(sun, sun.target);
  look.sun = sun;
  // A second, shadowless light: only on at night (the floodlights come from every side).
  look.fill = new THREE.DirectionalLight(0xffffff, 0);
  look.fill.target = sun.target;
  scene.add(look.fill);

  look.sky = noOutline(new THREE.Mesh(
    new THREE.SphereGeometry(camera.far * 0.9, 48, 24),
    new THREE.ShaderMaterial({ ...SKY_SHADER, side: THREE.BackSide, depthWrite: false, fog: false }),
  ));
  look.sky.renderOrder = -1;
  look.sky.frustumCulled = false;
  scene.add(look.sky);

  const size = renderer.getDrawingBufferSize(new THREE.Vector2());
  // Float32 so 1/z stays exact far out; nearest sampling (float textures need not filter).
  const nd = new THREE.WebGLRenderTarget(size.x, size.y, {
    type: THREE.FloatType, minFilter: THREE.NearestFilter, magFilter: THREE.NearestFilter,
  });
  const ndMaterial = makeNormalDepthMaterial(camera.far);
  // Ink-proof marker: a = -1 where a layer-2 object is in front (depth-tested against layer 0).
  const markMaterial = new THREE.ShaderMaterial({
    vertexShader: "void main() { gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }",
    fragmentShader: "void main() { gl_FragColor = vec4(0.0, 0.0, 0.0, -1.0); }",
    side: THREE.DoubleSide, blending: THREE.NoBlending, transparent: true,
  });
  const composer = new EffectComposer(renderer);
  composer.addPass(new RenderPass(scene, camera));
  look.edge = new ShaderPass(EDGE_SHADER);
  look.edge.uniforms.tND.value = nd.texture;
  look.edge.uniforms.uFar.value = camera.far;
  composer.addPass(look.edge);
  look.dof = new ShaderPass(DOF_SHADER);
  look.dof.uniforms.tND.value = nd.texture;
  look.dof.uniforms.uFar.value = camera.far;
  look.dof.enabled = false;
  composer.addPass(look.dof);
  // Threshold 1.2: only what is well brighter than white blooms (lit lamps x3, lit windows, the
  // sun disc), not sunlit chalk or the white strike zone.
  look.bloom = new UnrealBloomPass(new THREE.Vector2(size.x, size.y), 0.35, 0.5, 1.2);
  composer.addPass(look.bloom);
  composer.addPass(new SMAAPass());
  composer.addPass(new OutputPass());

  const clear = new THREE.Color();
  return {
    setSize(w, h) {
      composer.setPixelRatio(renderer.getPixelRatio());
      composer.setSize(w, h);
      const s = renderer.getDrawingBufferSize(new THREE.Vector2());
      nd.setSize(s.x, s.y);
      look.edge.uniforms.uTexel.value.set(1 / s.x, 1 / s.y);
      look.edge.uniforms.uScale.value = Math.max(1, s.y / 900);
      look.dof.uniforms.uTexel.value.set(1 / s.x, 1 / s.y);
    },
    /**
     * Lighter graphics: no shadow map (the sun stops casting; three recompiles the receivers) and no bloom.
     * Both off means the default look is untouched; the caller also drops the pixel ratio to 1.
     */
    setLite(on) {
      fanHook.lite = on; // park/crowd.js swaps to its simple fans
      look.sun.castShadow = !on;
      look.bloom.enabled = !on;
      if (!on) refreshShadows(); // the shadow map is only redrawn on request: it is stale after being off
    },
    /** Depth of field focused at focusFt (ft from the camera), blurring up to maxPx; null: off. */
    setFocus(focusFt, maxPx = 6) {
      look.dof.enabled = focusFt != null;
      if (focusFt == null) return;
      const s = renderer.getDrawingBufferSize(new THREE.Vector2());
      look.dof.uniforms.uFocus.value = focusFt;
      look.dof.uniforms.uMaxPx.value = maxPx * Math.max(1, s.y / 900);
    },
    render() {
      look.sky.position.copy(camera.position); // the sky is infinitely far: it follows the camera
      // 1. normals + depth of layer-0 objects
      const mask = camera.layers.mask;
      const alpha = renderer.getClearAlpha();
      renderer.getClearColor(clear);
      camera.layers.set(0);
      scene.overrideMaterial = ndMaterial;
      renderer.setRenderTarget(nd);
      renderer.setClearColor(0x000000, 1);
      renderer.clear();
      renderer.render(scene, camera);
      camera.layers.set(2);
      scene.overrideMaterial = markMaterial;
      const autoClear = renderer.autoClear;
      renderer.autoClear = false; // keep layer 0's depth so hidden parts of the ball stay unmarked
      renderer.render(scene, camera);
      renderer.autoClear = autoClear;
      scene.overrideMaterial = null;
      camera.layers.mask = mask;
      renderer.setClearColor(clear, alpha);
      renderer.setRenderTarget(null);
      // 2-5. scene, edges, SMAA, output
      composer.render();
    },
  };
}
