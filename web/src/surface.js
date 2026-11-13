/*
 * Painted surface texture, generated in the shader from world position (no image assets): every
 * toon surface gets mottling (stains, sun-bleaching) and fine grain, so flat colour reads as
 * painted plaster, wood, foliage, rock or ground, as in the hand-painted references (the
 * courtyard's worn stone, Messenger's textured walls). World space, not UVs: the swept stands and
 * instanced props have none, and the pattern then never stretches.
 *
 * Kinds (the SURFACE define, set by look.js makeToonMaterial):
 *   0 none; 1 plaster/concrete (mottling, grime at the foot, drip streaks, hairline cracks);
 *   2 foliage (big light/dark masses and ~2 ft leaf clusters); 3 wood (grain);
 *   4 ground (grass, clay, chalk: see groundSurface); 5 rock (strata, cracks).
 *
 * Inputs from look.js: vParkW (world position, ft; scene axes, so Statcast y = -z), uSunDir
 * (unit vector toward the key light, world), uUvPerFt (painted ground layers only: map uv per ft).
 */
export const SURFACE_KINDS = { none: 0, plaster: 1, foliage: 2, wood: 3, ground: 4, rock: 5 };

export const SURFACE_GLSL = /* glsl */`
  float sHash(vec3 p) { return fract(sin(dot(p, vec3(127.1, 311.7, 74.7))) * 43758.5453); }
  float sHash2(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
  float sNoise(vec3 p) {
    vec3 i = floor(p), f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    return mix(mix(mix(sHash(i), sHash(i + vec3(1, 0, 0)), f.x), mix(sHash(i + vec3(0, 1, 0)), sHash(i + vec3(1, 1, 0)), f.x), f.y),
               mix(mix(sHash(i + vec3(0, 0, 1)), sHash(i + vec3(1, 0, 1)), f.x), mix(sHash(i + vec3(0, 1, 1)), sHash(i + vec3(1, 1, 1)), f.x), f.y), f.z);
  }
  // Thin dark line where a noise field crosses 0.5: hairline cracks (1 on the crack, 0 off it).
  float crackLine(float n, float w) { return 1.0 - smoothstep(0.0, w, abs(n - 0.5)); }

  float surfaceTone(vec3 w) {
    // Few noise lookups per pixel: this runs on every toon pixel, under heavy tree overdraw.
    float big = sNoise(w * 0.09); // ~10 ft stains
    float fine = sNoise(w * 1.7); // ~0.6 ft grain
    #if SURFACE == 1
      float crack = 1.0 - 0.2 * crackLine(sNoise(w * vec3(0.21, 0.09, 0.21)), 0.007) * step(0.7, big);
      float grime = mix(0.86, 1.0, smoothstep(0.0, 7.0, w.y)); // splash and dirt at the foot of walls
      // Rain streaks: noise stretched ~25x along the vertical, so stains run down from ledges.
      float streak = 1.0 - 0.08 * smoothstep(0.55, 0.85, sNoise(vec3(w.x * 0.45, w.y * 0.018, w.z * 0.45)));
      return (0.9 + 0.18 * big + 0.07 * fine) * crack * grime * streak;
    #elif SURFACE == 2
      // Leaf clusters: ~2 ft clumps, lit domes with darker gaps between them (the painted-tree
      // look), over the big light/dark masses.
      float clump = smoothstep(0.32, 0.68, sNoise(w * 0.55));
      return (0.82 + 0.26 * big + 0.06 * fine) * mix(0.8, 1.1, clump);
    #elif SURFACE == 3
      float grain = sNoise(vec3(w.x * 0.4, w.y * 7.0, w.z * 0.4));
      return 0.84 + 0.2 * grain + 0.06 * fine;
    #elif SURFACE == 4
      return 0.92 + 0.12 * big + 0.06 * fine; // paving and other painted ground (see groundSurface)
    #elif SURFACE == 5
      // Sedimentary strata: noise squashed ~15x along the vertical gives wavy layers ~15 ft
      // thick (the island's underside is ~460 ft deep), warped by a broad noise, with vertical
      // joints (cracks) between blocks; boulders get mostly the fine grain.
      float rbig = sNoise(w * 0.012);
      float strata = sNoise(vec3(w.x * 0.004, w.y * 0.065, w.z * 0.004) + rbig * 0.8);
      float joint = 1.0 - 0.2 * crackLine(sNoise(vec3(w.x * 0.02, w.y * 0.004, w.z * 0.02)), 0.02);
      return (0.74 + 0.34 * strata + 0.1 * fine + 0.12 * big) * joint;
    #else
      return 1.0;
    #endif
  }

  #if SURFACE == 4
  float groundLight = 1.0; // sun bump factor, set by groundSurface, applied to direct light

  /*
   * The ground: grass, clay and chalk, told apart by the painted colour (linear rgb) so one shader
   * serves the painted layers, the mound, plate and rubber. All in Statcast feet, S = (x, y).
   * Detail at a scale finer than ~3 px is faded out by the pixel's footprint (fp, ft per pixel),
   * which stops it shimmering far away; each term averages ~1 so the fade keeps the tone.
   * It also writes a small relief height h (ft) whose slope tilts the sun's lighting (bump):
   * on near-flat ground, N.L changes by about -(grad h . L_xz) / L_y.
   */
  vec3 groundSurface(vec3 col, vec3 w) {
    vec2 S = vec2(w.x, -w.z);
    float fp = max(length(dFdx(w)), length(dFdy(w)));
    float nearFine = 1.0 - smoothstep(0.015, 0.06, fp); // ~0.15 ft detail: blades, grains, cleats
    float nearMid = 1.0 - smoothstep(0.06, 0.3, fp); // ~0.5 ft detail: clumps, rake lines
    float big = sNoise(w * 0.07); // ~14 ft: watering, wear, sun-bleach
    float patchy = sNoise(w * 0.21); // ~5 ft: damp/dry patches, thick/thin turf
    float mid = sNoise(w * 0.6); // ~1.7 ft clumps
    float fine = mix(0.5, sNoise(w * 7.0), nearFine); // ~0.14 ft grain

    float grassW = smoothstep(0.015, 0.06, col.g - max(col.r, col.b));
    float chalkW = smoothstep(0.55, 0.7, min(col.r, min(col.g, col.b)));
    // Clay is orange: little blue against red (paving and concrete have much more).
    float dirtW = (1.0 - grassW) * (1.0 - chalkW) * (1.0 - smoothstep(0.3, 0.45, col.b / max(col.r, 1e-3)));

    // Where feet go: the batter's boxes (the batter digs in near the back), the pitcher's landing
    // spot about 6 ft in front of the rubber and the push-off hole at its front edge.
    float inBox = 1.0 - smoothstep(0.0, 0.7, max(abs(abs(S.x) - 3.2) - 2.0, abs(S.y - 0.7) - 3.0));
    float dug = 1.0 - smoothstep(0.4, 1.5, length(vec2(abs(S.x) - 3.0, (S.y - 0.6) * 0.75)));
    float land = 1.0 - smoothstep(0.7, 2.2, length((S - vec2(0.0, 54.5)) * vec2(0.75, 1.0)));
    float push = 1.0 - smoothstep(0.5, 1.4, length((S - vec2(0.0, 60.1)) * vec2(0.6, 1.0)));
    float traffic = max(max(0.55 * inBox, dug), max(land, 0.8 * push));

    // --- clay: infield skin, home circle, mound, warning track ---
    vec3 d = col * (0.88 + 0.24 * big) * (0.9 + 0.2 * patchy) * (0.9 + 0.2 * fine);
    float hd = 0.015 * (mid - 0.5) + 0.012 * (fine - 0.5);
    // Pebbles and clods: at most one per 0.3 ft cell, some lighter (dry), some darker.
    vec2 pc = floor(S / 0.3), pf = fract(S / 0.3) - 0.5;
    float ph = sHash2(pc);
    vec2 po = vec2(sHash2(pc + 17.0), sHash2(pc + 41.0)) - 0.5;
    float peb = (1.0 - smoothstep(0.07, 0.14, length(pf - 0.55 * po))) * step(0.7, ph) * nearFine;
    d *= 1.0 + peb * (ph > 0.85 ? 0.22 : -0.2);
    hd += 0.02 * peb;
    // Drag lines: faint grooves ~5 in apart, concentric round the plate (home circle) and the
    // rubber (skin, warning track), wobbling, in patches (the drag mat's passes overlap and fade),
    // gone where the ground is walked on and on the mound (tamped, not dragged).
    vec2 rc = S.y < 32.0 ? vec2(0.0, 0.7) : vec2(0.0, 59.0);
    float rake = sin((length(S - rc) + 0.6 * mid + 0.15 * fine) * 6.2831853 / 0.42);
    float rakeAmt = smoothstep(0.45, 0.7, big) * smoothstep(0.3, 0.6, mid) * nearMid * (1.0 - traffic)
      * smoothstep(9.5, 11.0, length(S - vec2(0.0, 59.0)));
    d *= 1.0 - 0.035 * rake * rakeAmt;
    hd += 0.006 * rake * rakeAmt;
    // Damp clay: groundskeepers water round the plate and the mound; darker and a bit redder.
    float wet = max(1.0 - smoothstep(3.0, 11.0, length(S - vec2(0.0, 0.7))),
                    0.7 * (1.0 - smoothstep(3.0, 9.0, length(S - vec2(0.0, 59.0)))));
    wet *= smoothstep(0.3, 0.65, 0.5 * (big + mid));
    d *= mix(vec3(1.0), vec3(0.8, 0.74, 0.74), wet);
    // Cleat marks: short gouges at random angles, one per 0.35 ft cell, denser with traffic.
    vec2 cc = floor(S / 0.35), cf = fract(S / 0.35) - 0.5;
    float ca = sHash2(cc + 3.1) * 6.2831853;
    vec2 cd = vec2(cos(ca), sin(ca));
    float gouge = (1.0 - smoothstep(0.025, 0.05, abs(dot(cf, vec2(-cd.y, cd.x)))))
      * (1.0 - smoothstep(0.1, 0.18, abs(dot(cf, cd))))
      * step(1.0 - 0.75 * traffic, sHash2(cc + 9.7)) * nearFine;
    d *= 1.0 - 0.3 * gouge;
    hd -= 0.03 * gouge;
    // Footprints: the catcher, umpire and batters walk all over the home circle. At most one
    // print per 1.4 ft cell, likelier near the plate: a sole and a heel (two ellipses, ~11 in
    // long in all) pressed in, at a random heading.
    vec2 fc = floor(S / 1.4), ff = (fract(S / 1.4) - 0.5) * 1.4;
    float fh = sHash2(fc + 5.3);
    float fa = sHash2(fc + 8.9) * 6.2831853;
    vec2 fu = vec2(cos(fa), sin(fa)), fq = ff - 0.4 * (vec2(sHash2(fc + 2.2), fh) - 0.5);
    vec2 fl = vec2(dot(fq, fu), dot(fq, vec2(-fu.y, fu.x)));
    float sole = length((fl - vec2(0.13, 0.0)) / vec2(0.27, 0.15)), heel = length((fl + vec2(0.3, 0.0)) / vec2(0.13, 0.12));
    float walk = max(1.0 - smoothstep(4.0, 14.0, length(S - vec2(0.0, -1.0))), 0.6 * land);
    float print = (1.0 - smoothstep(0.75, 1.0, min(sole, heel))) * step(1.0 - 0.4 * walk, fh) * nearMid;
    d *= 1.0 - 0.08 * print;
    hd -= 0.01 * print;
    // Churned clay where feet go: lumpier, light kicked-up dust over darker dug-in spots.
    d *= 1.0 + traffic * (0.3 * (mid - 0.5) + 0.15 * (fine - 0.5));
    hd += traffic * 0.03 * (mid - 0.5);

    // --- grass ---
    // Colour drift: big patches toward straw yellow (dry, worn) or blue-green (well watered).
    vec3 g = col * mix(vec3(1.07, 1.03, 0.8), vec3(0.9, 1.0, 1.08), big);
    g *= (0.88 + 0.24 * patchy) * (0.9 + 0.2 * mid) * (0.82 + 0.36 * fine);
    float hg = 0.02 * (mid - 0.5) + 0.03 * (fine - 0.5);
    // Mowing sheen: blades in a stripe lean the way the mower went. Looking along the lean you see
    // their lit sides (lighter), against it their darker tips and shade, so the painted stripes
    // strengthen seen from home and invert seen from centre field, as reverse-angle broadcasts
    // show; strongest at grazing angles. Stripes as painted in park/ground.js: 15 ft bands across
    // the centre-field line, and an 11.25 ft checkerboard turned 45 deg on the infield grass.
    vec2 ab = vec2(S.x + S.y, S.y - S.x) * 0.70710678;
    bool infield = min(ab.x, ab.y) > 0.0 && max(ab.x, ab.y) < 90.0;
    float sgn = infield ? 1.0 - 2.0 * mod(floor(ab.x / 11.25) + floor(ab.y / 11.25), 2.0)
                        : 1.0 - 2.0 * mod(floor(S.y / 15.0), 2.0);
    float striped = infield || abs(S.x) < S.y ? 1.0 : 0.0;
    vec3 V = normalize(w - cameraPosition);
    float along = -V.z / max(length(V.xz), 1e-3); // view direction along Statcast +y
    g *= 1.0 + 0.09 * sgn * along * striped * (1.0 - 0.6 * abs(V.y));

    // --- chalk: a little rough, and scuffed through to the clay where feet go ---
    float scuff = smoothstep(0.62, 0.9, 0.45 * mid + 0.35 * fine + 0.5 * traffic);
    vec3 c = mix(col * (0.94 + 0.08 * fine), vec3(0.584, 0.242, 0.102) * (0.9 + 0.2 * mid), 0.85 * scuff);

    vec3 o = col * (0.92 + 0.12 * big + 0.06 * fine); // paving, paths and the rest
    o = mix(o, d, dirtW);
    o = mix(o, g, grassW);
    o = mix(o, c, chalkW);
    float h = mix(mix(0.0, hd, dirtW), hg, grassW);
    // Small crevices (gouges, gaps between clods and blades) also miss sky light.
    o *= clamp(1.0 + 4.0 * h, 0.7, 1.15);

    // Bump: solve for the world xz gradient of h from its screen derivatives (2x2 system).
    vec2 dx = vec2(dFdx(w.x), dFdx(w.z)), dy = vec2(dFdy(w.x), dFdy(w.z));
    float hx = dFdx(h), hy = dFdy(h);
    float det = dx.x * dy.y - dx.y * dy.x;
    vec2 grad = abs(det) > 1e-10 ? vec2(hx * dy.y - dx.y * hy, dx.x * hy - hx * dy.x) / det : vec2(0.0);
    // (The 1 / L_y is capped at 2: under a low sun the true factor would blow every clod and
    // footprint up into a crater.)
    groundLight = clamp(1.0 - dot(grad, uSunDir.xz) / max(uSunDir.y, 0.5), 0.6, 1.4);
    return o;
  }
  #endif
`;
