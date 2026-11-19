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
 * Inputs from look.js: vParkW (world position, ft; scene axes, so Statcast y = -z) and uSunDir
 * (unit vector toward the key light, world). CHALK_CLEAN (define): chalk never scuffed toward
 * clay (home plate, the rubber).
 */
import { chalkSegments, CHALK_WIDTH_FT } from "./park/layout.js";
import { isOld } from "./park/old.js";

/*
 * Turf v2 (?old=grass restores v1), added after the grass read as flat lime:
 * - colour: the painted lime pulled toward a deeper broadcast green (less red, a touch of blue);
 * - stripes: a fixed light/dark step on top of the view-dependent sheen, so the mowing pattern
 *   reads from every camera (the painted stripes alone differ by ~10%, near-invisible from TV);
 * - blades: up close, fine streaks stretched along the mower's direction (~0.1 ft wide, ~0.7 ft
 *   long), posterised into lit blades and darker gaps, plus a matching bump; faded by pixel
 *   footprint so it never shimmers far away.
 */
const TURF_V2 = !isOld("grass");

export const SURFACE_KINDS = { none: 0, plaster: 1, foliage: 2, wood: 3, ground: 4, rock: 5 };

// The chalk lines as GLSL constants: segment ends (xy, zw, Statcast ft) and kind.
const SEGS = chalkSegments();
const f4 = (v) => v.toFixed(4);
const CHALK_GLSL = `
  const int CHALK_N = ${SEGS.length};
  const vec4 CHALK_AB[${SEGS.length}] = vec4[${SEGS.length}](${SEGS.map(([a, b]) => `vec4(${[...a, ...b].map(f4).join(", ")})`).join(", ")});
  const float CHALK_KIND[${SEGS.length}] = float[${SEGS.length}](${SEGS.map((x) => `${x[2]}.0`).join(", ")});
  const float CHALK_HALF_W = ${f4(CHALK_WIDTH_FT / 2)};`;

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
  ${CHALK_GLSL}
  float groundLight = 1.0; // sun bump factor, set by groundSurface, applied to direct light

  // Distance (ft) from p to the segment a-b.
  float segDist(vec2 p, vec2 a, vec2 b) {
    vec2 ab = b - a;
    return length(p - a - ab * clamp(dot(p - a, ab) / dot(ab, ab), 0.0, 1.0));
  }
  // A soft two-tone step: 0 below e, 1 above, with a narrow painted edge (not a gradient).
  float band(float v, float e) { return smoothstep(e - 0.025, e + 0.025, v); }

  /*
   * The ground, stylised: real MLB infield structure painted in grouped tones, not photographic
   * noise (Ryan, 2026-11-13: realism "within theme"). Grass, clay and chalk are told apart by the
   * painted colour (linear rgb), so one shader serves the painted layers, the mound, plate and
   * rubber, and every dirt area (home circle, base paths to first and third, the skin round the
   * bases, mound, warning track) gets the same treatment. All in Statcast feet, S = (x, y).
   *
   * Clay: three value groups (dry light top, mid, darker damp clay) laid out as soft painted
   * blotches; wide drag arcs from the groundskeeper's mat; the runners' paths and the bases
   * scuffed; a fine sandy grain quantised to three tones. Grass: mowing stripes with
   * view-dependent sheen, soft posterised patches, two-tone clumps. Detail finer than ~3 px is
   * faded by the pixel footprint (fp, ft per pixel). A small relief height h (ft) tilts the sun's
   * lighting (bump): on near-flat ground, N.L changes by about -(grad h . L_xz) / L_y.
   */
  vec3 groundSurface(vec3 col, vec3 w) {
    vec2 S = vec2(w.x, -w.z);
    float fp = max(length(dFdx(w)), length(dFdy(w)));
    float nearFine = 1.0 - smoothstep(0.015, 0.06, fp); // ~0.15 ft: sand grain, tufts
    float nearMid = 1.0 - smoothstep(0.06, 0.3, fp); // ~0.5 ft: clumps, footprints
    float big = sNoise(w * 0.07); // ~14 ft
    // ~5 ft, domain-warped by the big noise so blotches get uneven, brush-like outlines.
    float patchy = sNoise(w * 0.21 + vec3(2.0 * big));
    float mid = sNoise(w * 0.6); // ~1.7 ft
    float fine = mix(0.5, sNoise(w * 7.0), nearFine); // ~0.14 ft

    // Grass is clearly green: green over red and blue, and well over blue (so the teal on-deck
    // circles and the water are not grass).
    float grassW = smoothstep(0.015, 0.06, col.g - max(col.r, col.b)) * smoothstep(1.3, 1.8, col.g / max(col.b, 1e-3));
    float chalkW = smoothstep(0.55, 0.7, min(col.r, min(col.g, col.b)));
    // Clay is orange: little blue against red (paving and concrete have much more).
    float dirtW = (1.0 - grassW) * (1.0 - chalkW) * (1.0 - smoothstep(0.3, 0.45, col.b / max(col.r, 1e-3)));

    // Where feet go: the batter's boxes (the batter digs in near the back), the pitcher's landing
    // spot ~6 ft in front of the rubber and the push-off hole, the runners' paths between the
    // bases (OBR 2.01: 90 ft apart) and round each base, where runners round, slide and stand.
    float inBox = 1.0 - smoothstep(0.0, 0.7, max(abs(abs(S.x) - 3.2) - 2.0, abs(S.y - 0.7) - 3.0));
    float dug = 1.0 - smoothstep(0.4, 1.5, length(vec2(abs(S.x) - 3.0, (S.y - 0.6) * 0.75)));
    float land = 1.0 - smoothstep(0.7, 2.2, length((S - vec2(0.0, 54.5)) * vec2(0.75, 1.0)));
    float push = 1.0 - smoothstep(0.5, 1.4, length((S - vec2(0.0, 60.1)) * vec2(0.6, 1.0)));
    vec2 B1 = vec2(63.64, 63.64), B2 = vec2(0.0, 127.28), B3 = vec2(-63.64, 63.64);
    float path = min(min(segDist(S, vec2(0.0), B1), segDist(S, B1, B2)), min(segDist(S, B2, B3), segDist(S, B3, vec2(0.0))));
    float base = min(min(length(S - B1), length(S - B2)), length(S - B3));
    float runway = (1.0 - smoothstep(1.0, 2.8, path)) * smoothstep(12.0, 16.0, length(S)); // not in the home circle
    float baseWear = 1.0 - smoothstep(2.0, 6.0, base);
    float traffic = max(max(0.55 * inBox, dug), max(max(land, 0.8 * push), max(0.7 * runway, baseWear)));

    // --- clay ---
    // Three value groups. v: soft blotches (14 ft, and warped 5 ft), raised toward damp where the
    // groundskeepers water (round the plate and the mound) and in the dug spots.
    float wet = max(1.0 - smoothstep(3.0, 12.0, length(S - vec2(0.0, 0.7))),
                    0.8 * (1.0 - smoothstep(3.0, 10.0, length(S - vec2(0.0, 59.0)))));
    float v = 0.55 * big + 0.45 * patchy + 0.3 * wet + 0.2 * dug - 0.1;
    vec3 dry = col * vec3(1.1, 1.07, 1.02), damp = col * vec3(0.8, 0.69, 0.64);
    vec3 d = mix(mix(dry, col, band(v, 0.36)), damp, band(v, 0.6));
    // Drag arcs: the mat is pulled in wide loops round the infield, leaving ~3 ft bands of
    // slightly lighter and darker clay, concentric on the rubber (round the plate in the home
    // circle); broken up by the blotches, gone on the mound and where feet go.
    vec2 rc = S.y < 32.0 ? vec2(0.0, 0.7) : vec2(0.0, 59.0);
    float arc = band(0.5 + 0.5 * sin((length(S - rc) + 1.5 * patchy) * 6.2831853 / 6.0), 0.5);
    float dragAmt = (1.0 - traffic) * smoothstep(10.0, 12.0, length(S - vec2(0.0, 59.0))) * band(patchy, 0.4);
    d *= 1.0 + 0.05 * (arc - 0.5) * dragAmt;
    // Scuffed clay where feet go: kicked-up dry topsoil in grouped light patches, with darker
    // dug-in spots between them.
    float kick = band(mid, 0.5);
    d = mix(d, mix(damp * 1.05, dry * 1.04, kick), 0.65 * traffic);
    // Footprints round home, the landing spot and the bases: soft sole-and-heel shapes, at most
    // one per 1.4 ft cell.
    vec2 fc = floor(S / 1.4), ff = (fract(S / 1.4) - 0.5) * 1.4;
    float fh = sHash2(fc + 5.3);
    float fa = sHash2(fc + 8.9) * 6.2831853;
    vec2 fu = vec2(cos(fa), sin(fa)), fq = ff - 0.4 * (vec2(sHash2(fc + 2.2), fh) - 0.5);
    vec2 fl = vec2(dot(fq, fu), dot(fq, vec2(-fu.y, fu.x)));
    float sole = length((fl - vec2(0.13, 0.0)) / vec2(0.27, 0.15)), heel = length((fl + vec2(0.3, 0.0)) / vec2(0.13, 0.12));
    float walk = max(1.0 - smoothstep(4.0, 12.0, length(S - vec2(0.0, -1.0))), max(0.6 * land, 0.5 * baseWear));
    float print = (1.0 - smoothstep(0.8, 0.95, min(sole, heel))) * step(1.0 - 0.35 * walk, fh) * nearMid;
    d *= 1.0 - 0.07 * print;
    // Fine sand: the grain quantised to three tones (a few light grains, a few dark), not a
    // continuous static; plus a sparse scatter of lighter grit.
    float grain = band(fine, 0.64) - (1.0 - band(fine, 0.36));
    float sandNear = 1.0 - smoothstep(0.006, 0.025, fp); // ~0.06 ft sand, only up close
    float sand = mix(0.5, sNoise(w * 16.0 + 7.0), sandNear);
    d *= (1.0 + 0.05 * grain) * (1.0 + 0.05 * (band(sand, 0.62) - (1.0 - band(sand, 0.38))));
    vec2 pc = floor(S / 0.3), pf = fract(S / 0.3) - 0.5;
    float ph = sHash2(pc);
    vec2 po = vec2(sHash2(pc + 17.0), sHash2(pc + 41.0)) - 0.5;
    float peb = (1.0 - smoothstep(0.06, 0.1, length(pf - 0.55 * po))) * step(0.8, ph) * nearFine;
    d *= 1.0 + 0.12 * peb;
    float hd = 0.012 * (mid - 0.5) + 0.004 * (arc - 0.5) * dragAmt + 0.01 * traffic * (kick - 0.5) - 0.008 * print;

    // --- grass ---
    // Colour drift: big soft areas toward straw yellow (dry) or blue-green (well watered).
    vec3 g = col * mix(vec3(1.06, 1.03, 0.84), vec3(0.92, 1.0, 1.06), smoothstep(0.3, 0.7, big));
    // Value patches posterised into three soft steps, then ~1.7 ft clumps in two tones and a few
    // lighter and darker tufts at blade scale.
    float gv = 0.55 * big + 0.45 * patchy;
    g *= 0.93 + 0.06 * band(gv, 0.42) + 0.06 * band(gv, 0.6);
    float clump = band(mid, 0.5);
    g *= 1.0 + 0.06 * (clump - 0.5) * nearMid;
    g *= 1.0 + 0.05 * (band(fine, 0.68) - (1.0 - band(fine, 0.32)));
    float hg = 0.012 * (clump - 0.5) * nearMid;
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
    // v2: abs(along). The true reversal cancelled the painted stripes from centre field (TV view);
    // here every camera sees them strengthened (a stylised choice, not the physical sheen).
    g *= 1.0 + 0.09 * sgn * ${TURF_V2 ? "abs(along)" : "along"} * striped * (1.0 - 0.6 * abs(V.y));
    ${TURF_V2 ? `
    // Broadcast turf: deeper and a touch blue. Tuned on screen (after tone mapping) toward
    // ~rgb(75,130,48) from the old lime ~rgb(123,181,60).
    g *= vec3(0.5, 0.68, 0.56);
    g *= 1.0 + 0.05 * sgn * striped;
    // Mower direction: across the centre-field line (along x) in the outfield; along the
    // checkerboard's diagonals on the infield. Streaks are stretched along it.
    vec2 mw = infield ? ab : S;
    float bladeVis = 1.0 - smoothstep(0.02, 0.12, fp);
    float blade = band(sNoise(vec3(mw.x * 1.4, 3.0, mw.y * 10.0)), 0.52);
    g *= 1.0 + 0.11 * (blade - 0.5) * bladeVis;
    hg += 0.006 * (blade - 0.5) * bladeVis;` : ""}

    // --- chalk: a little rough, and scuffed through to the clay where feet go; never on the
    // plate or the rubber (CHALK_CLEAN) ---
    #ifdef CHALK_CLEAN
      float scuff = 0.0;
    #else
      // Cleats scuff the lines in the boxes, at the plate end and round the bases.
      float scuff = band(0.45 * mid + 0.35 * fine + 0.5 * max(max(0.55 * inBox, dug), 0.7 * baseWear), 0.78);
    #endif
    vec3 c = mix(col * (0.97 + 0.03 * grain), vec3(0.584, 0.242, 0.102) * (0.95 + 0.1 * kick), 0.85 * scuff);

    vec3 o = col * (0.95 + 0.07 * band(big, 0.5)); // paving, paths and the rest
    o = mix(o, d, dirtW);
    o = mix(o, g, grassW);
    o = mix(o, c, chalkW);
    float h = mix(mix(0.0, hd, dirtW), hg, grassW);

    // --- chalk lines, drawn analytically (park/layout.js chalkSegments), 3 in wide ---
    // Nearest distance (ft) to each kind of line: foul, batter's box, catcher's box, other.
    vec4 dk = vec4(1e3);
    for (int i = 0; i < CHALK_N; i++) {
      float dd = segDist(S, CHALK_AB[i].xy, CHALK_AB[i].zw);
      float k = CHALK_KIND[i];
      if (k < 0.5) dk.x = min(dk.x, dd);
      else if (k < 1.5) dk.y = min(dk.y, dd);
      else if (k < 2.5) dk.z = min(dk.z, dd);
      else dk.w = min(dk.w, dd);
    }
    float dmin = min(min(dk.x, dk.y), min(dk.z, dk.w));
    // Smooth, continuous lines first. The edge is anti-aliased by the footprint of the distance
    // itself (fwidth(dmin): exactly how far it changes across one pixel, so lines seen edge-on at
    // the umpire's grazing angle are not over-blurred along their length), never a hard step: a
    // line narrower than a pixel far away keeps about its true area and fades instead of breaking
    // into stair-steps. Up close the edge is a little powdery on clay (~0.4 in soft) and crisper
    // on grass, where lines are sprayed onto the blades. Width and density vary only slowly along
    // a line (~6 ft and ~14 ft noise), as a liner's do; nothing at speckle scale.
    float hw = CHALK_HALF_W * (1.0 + 0.12 * (patchy - 0.5));
    float soft = max(0.75 * fwidth(dmin), mix(0.008, 0.025, dirtW));
    float cov = 1.0 - smoothstep(hw - soft, hw + soft, dmin);
    float slow = sNoise(w * 0.16 + 3.0); // ~6 ft along the lines
    cov *= 0.9 + 0.1 * slow;
    // Wear from a game, as a few long, gentle fades and smudges, never gaps:
    // - batter's boxes: faded where the hitter's feet go (mostly the inner and back lines); the
    //   outer and front lines stay, so each box still reads as a box;
    // - foul lines: faded a little where players cross them near the plate and the bases, more
    //   on clay than on grass;
    // - catcher's box: faint all over (rarely relined during a game).
    // Which line a pixel belongs to, softly (no step where a foul line meets a box corner).
    float isBox = 1.0 - smoothstep(0.0, 0.2, dk.y - dmin), isFoul = 1.0 - smoothstep(0.0, 0.2, dk.x - dmin);
    float isCatcher = 1.0 - smoothstep(0.0, 0.2, dk.z - dmin);
    float feet = 1.0 - smoothstep(0.8, 2.8, length(vec2(abs(S.x) - 2.6, (S.y - 0.4) * 0.6)));
    float boxWear = 0.7 * feet * smoothstep(0.2, 0.8, 0.65 * slow + 0.35 * big);
    // Players cross the foul lines a few feet past the boxes (starting at zero where the line
    // leaves the box, so the join stays clean) and round the bases.
    float crossing = max(smoothstep(5.5, 8.0, length(S)) * (1.0 - smoothstep(9.0, 13.0, length(S))), baseWear);
    float foulWear = 0.45 * crossing * smoothstep(0.3, 0.8, slow) * mix(0.5, 1.0, dirtW);
    float wear = max(isBox * boxWear, isFoul * foulWear);
    float onGround = clamp(dirtW + grassW, 0.0, 1.0); // not on the plate, rubber or paving
    cov *= (1.0 - wear) * mix(1.0, 0.5 + 0.15 * slow, isCatcher) * onGround;
    // Chalk dust: a faint powder a few inches round the lines on clay, a little more where a line
    // has been scuffed and smeared.
    float dust = (1.0 - smoothstep(hw, hw + 0.3, dmin)) * (0.06 + 0.12 * wear) * smoothstep(0.3, 0.7, mid);
    dust *= dirtW * onGround * nearMid;
    o = mix(o, vec3(0.92, 0.88, 0.77), clamp(max(cov, dust), 0.0, 1.0));

    // Bump: solve for the world xz gradient of h from its screen derivatives (2x2 system).
    vec2 dx = vec2(dFdx(w.x), dFdx(w.z)), dy = vec2(dFdy(w.x), dFdy(w.z));
    float hx = dFdx(h), hy = dFdy(h);
    float det = dx.x * dy.y - dx.y * dy.x;
    vec2 grad = abs(det) > 1e-10 ? vec2(hx * dy.y - dx.y * hy, dx.x * hy - hx * dy.x) / det : vec2(0.0);
    // (The 1 / L_y is capped at 2: under a low sun the true factor would blow every clod up.)
    groundLight = clamp(1.0 - dot(grad, uSunDir.xz) / max(uSunDir.y, 0.5), 0.7, 1.3);
    return o;
  }
  #endif
`;
