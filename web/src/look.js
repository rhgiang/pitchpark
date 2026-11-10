import * as THREE from "three";
import { EffectComposer } from "three/examples/jsm/postprocessing/EffectComposer.js";
import { RenderPass } from "three/examples/jsm/postprocessing/RenderPass.js";
import { ShaderPass } from "three/examples/jsm/postprocessing/ShaderPass.js";
import { SMAAPass } from "three/examples/jsm/postprocessing/SMAAPass.js";
import { OutputPass } from "three/examples/jsm/postprocessing/OutputPass.js";

/*
 * The look: one palette, toon materials, outlines, sky and time of day. Everything visible is
 * coloured from PALETTE, so the whole scene stays in one tight set of colours (the thing both
 * references do: Messenger's teal/cream/coral, the cricket park's red/white/green).
 *
 * Pipeline per frame (render()):
 *   1. Normal + depth pass: the outlined objects (layer 0) drawn with one override material that
 *      writes the view-space normal (rgb) and view depth (a) to a float target.
 *   2. RenderPass: the scene as usual (toon materials, fog, sky dome, shadows).
 *   3. Edge pass: ink where the depth or the normal changes abruptly (see EDGE_SHADER).
 *   4. SMAA (Jimenez et al. 2012): post-process anti-aliasing; MSAA would be lost on the
 *      composer's render targets anyway, and SMAA also smooths the drawn ink lines.
 *   5. OutputPass: linear -> sRGB (no tone mapping: the palette is meant to come out as written).
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

function addFloodFalloff(shader) {
  shader.uniforms.uFlood = flood;
  shader.uniforms.uFloodMin = floodMin;
  shader.vertexShader = patch(shader.vertexShader,
    ["#include <common>", "#include <common>\nvarying vec2 vParkXZ;"],
    ["#include <project_vertex>", `#include <project_vertex>
      vec4 parkW = vec4(transformed, 1.0);
      #ifdef USE_INSTANCING
        parkW = instanceMatrix * parkW;
      #endif
      vParkXZ = (modelMatrix * parkW).xz;`]);
  shader.fragmentShader = patch(shader.fragmentShader,
    ["#include <common>", "#include <common>\nvarying vec2 vParkXZ;\nuniform vec4 uFlood;\nuniform float uFloodMin;"],
    ["#include <lights_fragment_end>", `#include <lights_fragment_end>
      reflectedLight.directDiffuse *= mix(1.0, uFloodMin,
        smoothstep(uFlood.z, uFlood.w, distance(vParkXZ, uFlood.xy)));`]);
}

/**
 * Lighting numbers that depend on the park, in Statcast feet; any omitted keep their value.
 * centreFt: middle of the bowl (night falloff centre, sun aim); floodRadiiFt: [r0, r1], full
 * floodlight inside r0, faded out by r1; shadowHalfFt: half-width of the sun's shadow box.
 * Works before or after initLook.
 */
export function setParkLighting(opts = {}) {
  for (const k of ["centreFt", "floodRadiiFt", "shadowHalfFt"]) if (opts[k] !== undefined) parkLight[k] = opts[k];
  const [x, y] = parkLight.centreFt;
  flood.value.set(x, -y, ...parkLight.floodRadiiFt);
  if (!look.sun) return; // initLook reads parkLight when it runs
  look.sun.target.position.set(x, 0, -y);
  const h = parkLight.shadowHalfFt;
  Object.assign(look.sun.shadow.camera, { left: -h, right: h, top: h, bottom: -h });
  look.sun.shadow.camera.updateProjectionMatrix();
  if (look.timeOfDay) setTimeOfDay(look.timeOfDay); // re-place the sun about the new centre
}

/**
 * A toon material in the park's style: the shared two-tone ramp, fog, the night falloff.
 * @param color hex or THREE.Color (use PALETTE)
 * @param extra other MeshToonMaterial parameters (map, vertexColors, side, ...)
 */
export function makeToonMaterial(color, extra = {}) {
  const m = new THREE.MeshToonMaterial({ color, gradientMap: toonRamp(), ...extra });
  m.onBeforeCompile = addFloodFalloff;
  return m;
}

// Lamps on the light towers: pale by day, glowing at night (setTimeOfDay switches them).
export const lampMaterial = new THREE.MeshBasicMaterial({ color: PALETTE.lampOff });
export const glowMaterial = new THREE.SpriteMaterial({
  color: 0xfff1c8, transparent: true, blending: THREE.AdditiveBlending, depthWrite: false,
  opacity: 0, fog: false,
});

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

/** Layer 1 only: drawn, but not outlined (sky, glows, the strike zone, the ball, nets). */
export function noOutline(obj) {
  obj.traverse((o) => o.layers.set(1));
  return obj;
}

// ---------- outlines ----------

// Normal + depth for the edge pass. Writes view-space normal and view depth / far.
// transparent + NoBlending: keeps three from forcing alpha to 1 (it does for opaque materials).
function makeNormalDepthMaterial(far) {
  const m = new THREE.MeshNormalMaterial({ side: THREE.DoubleSide, transparent: true, blending: THREE.NoBlending });
  m.onBeforeCompile = (s) => {
    s.uniforms.uFar = { value: far };
    s.vertexShader = patch(s.vertexShader,
      ["#include <common>", "#include <common>\nvarying float vViewZ;"],
      ["#include <project_vertex>", "#include <project_vertex>\nvViewZ = -mvPosition.z;"]);
    s.fragmentShader = patch(s.fragmentShader,
      ["#include <packing>", "#include <packing>\nvarying float vViewZ;\nuniform float uFar;"],
      ["gl_FragColor = vec4( packNormalToRGB( normal ), diffuseColor.a );",
        "gl_FragColor = vec4( normal, vViewZ / uFar );"]);
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
    uFade: { value: new THREE.Vector2(180, 1100) }, // ft: full lines until 180, gone by 1100
  },
  vertexShader: /* glsl */`
    varying vec2 vUv;
    void main() { vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`,
  fragmentShader: /* glsl */`
    uniform sampler2D tDiffuse, tND;
    uniform vec2 uTexel, uFade;
    uniform vec3 uInk;
    uniform float uFar, uScale;
    varying vec2 vUv;
    vec4 c0; // the centre sample; an ink-proof neighbour (a < 0) reads as the centre
    vec4 nd(vec2 o) { vec4 v = texture2D(tND, vUv + o * uTexel * uScale); return v.a < 0.0 ? c0 : v; }
    float inv(float a) { return 1.0 / max(a, 1e-6); }
    // Near-side silhouette / ridge strength at step k px along both axes.
    float ridge(float iz, float k) {
      float sx = inv(nd(vec2(-k, 0.0)).a) + inv(nd(vec2(k, 0.0)).a) - 2.0 * iz;
      float sy = inv(nd(vec2(0.0, -k)).a) + inv(nd(vec2(0.0, k)).a) - 2.0 * iz;
      return max(-sx, -sy) / iz;
    }
    void main() {
      vec3 col = texture2D(tDiffuse, vUv).rgb;
      c0 = texture2D(tND, vUv);
      if (c0.a < 0.0) { gl_FragColor = vec4(col, 1.0); return; } // ball / ghost / zone: no ink
      vec4 c = c0;
      if (c.a >= 1.0) { gl_FragColor = vec4(col, 1.0); return; } // sky: no line
      float iz = inv(c.a);
      float depthEdge = smoothstep(0.02, 0.06, max(ridge(iz, 1.0), ridge(iz, 2.0)));
      vec4 l = nd(vec2(-1.0, 0.0)), r = nd(vec2(1.0, 0.0)), d = nd(vec2(0.0, -1.0)), u = nd(vec2(0.0, 1.0));
      // Normals only compared with neighbours on the same surface patch (not across the sky).
      float nDiff = 0.0;
      if (max(max(l.a, r.a), max(d.a, u.a)) < 1.0)
        nDiff = max(max(1.0 - dot(c.xyz, l.xyz), 1.0 - dot(c.xyz, r.xyz)),
                    max(1.0 - dot(c.xyz, d.xyz), 1.0 - dot(c.xyz, u.xyz)));
      float normalEdge = smoothstep(0.15, 0.35, nDiff);
      float edge = max(depthEdge, normalEdge) * (1.0 - smoothstep(uFade.x, uFade.y, c.a * uFar));
      // Ink takes a little of the colour under it (as hand-inked lines do), so lines on dark
      // surfaces stay dark and lines on cream read as warm brown-grey.
      vec3 ink = mix(uInk, col * 0.3, 0.3);
      gl_FragColor = vec4(mix(col, ink, edge), 1.0);
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
    uCloudCover: { value: 0.55 }, uStars: { value: 0 },
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
    uniform float uSunSize, uCloudCover, uStars;
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
      col = mix(col, uSunColor, step(1.0 - uSunSize, s));
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
      float star = step(0.985, hash(sg)) * step(length(fract(sp) - 0.5), 0.22) // a round dot per lit cell
        * smoothstep(0.05, 0.3, h) * (1.0 - cloud);
      col += uStars * star * 0.9;
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
  },
  sunset: {
    top: 0x4a3b80, mid: 0xc8708f, horizon: 0xffbe7d, below: 0xf2a27e,
    cloud: 0xffc6a6, cloudShade: 0xa9628f, cover: 0.5, stars: 0,
    // Setting over centre field: low sun behind the pitcher, so the bowl faces into it (a sun
    // behind home would put the whole infield in the stands' shadow).
    sun: [-80, 10], sunColor: 0xffb070, sunIntensity: 2.6, sunSize: 0.0016,
    sky: 0xf0b4c0, ground: 0xb08070, ambient: 1.5,
    fog: 0xf0a47e, fogDensity: 0.0003, ink: 0x3a2433, floodMin: 1, lamps: true,
  },
  night: {
    top: 0x070d22, mid: 0x14204a, horizon: 0x2b3f6e, below: 0x1a2648,
    cloud: 0x2c3c66, cloudShade: 0x1b264a, cover: 0.62, stars: 1,
    sun: [-90, 62], sunColor: 0xf0f4ff, sunIntensity: 1.9, sunSize: 0.0009, // key: from centre field, high
    sky: 0x5a6ea8, ground: 0x1c2438, ambient: 0.5,
    fog: 0x1f2d55, fogDensity: 0.0003, ink: 0x05070d, floodMin: 0.05, lamps: true,
    fill: [90, 45], fillIntensity: 1.1, // floodlights from behind home too, so faces toward the plate aren't black
  },
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
  const T = TIMES[name];
  const u = look.sky.material.uniforms;
  u.uTop.value.set(T.top); u.uMid.value.set(T.mid);
  u.uHorizon.value.set(T.horizon); u.uBelow.value.set(T.below);
  u.uCloud.value.set(T.cloud); u.uCloudShade.value.set(T.cloudShade);
  u.uCloudCover.value = T.cover; u.uStars.value = T.stars;
  u.uSunDir.value.copy(dirFrom(name === "night" ? MOON : T.sun));
  u.uSunColor.value.set(name === "night" ? 0xeef2ff : T.sunColor);
  u.uSunSize.value = T.sunSize;
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
  lampMaterial.color.set(T.lamps ? PALETTE.lampOn : PALETTE.lampOff);
  glowMaterial.opacity = name === "night" ? 0.7 : T.lamps ? 0.35 : 0;
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
 * @returns {{render: () => void, setSize: (w: number, h: number) => void}}
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
