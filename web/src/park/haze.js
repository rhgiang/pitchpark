/**
 * Distance haze for the stands (?old=structure removes it): beyond ~240 ft a material fades toward
 * the fog colour, up to 38% by 950 ft, in four flat steps (not a smooth ramp, so it stays toon). The
 * scene's own fog is far too thin to show across a ballpark (FogExp2 at 0.0002 is 1% at 500 ft), and
 * the far stands read as near without it. Call on a material before its first render.
 */
export function addHaze(material, strength = 0.38) {
  const prev = material.onBeforeCompile;
  material.onBeforeCompile = (s, r) => {
    prev?.(s, r);
    s.fragmentShader = s.fragmentShader.replace("#include <fog_fragment>", `#include <fog_fragment>
      #ifdef USE_FOG
        gl_FragColor.rgb = mix(gl_FragColor.rgb, fogColor, floor(smoothstep(240.0, 950.0, vFogDepth) * 4.0) / 4.0 * ${strength.toFixed(2)});
      #endif`);
  };
  material.customProgramCacheKey = () => `haze${strength}`;
  return material;
}
