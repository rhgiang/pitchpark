/*
 * When to draw (GPU load). Measured 2026-11-14 on an RTX 5060 Ti at 2560x1440: one frame (shadow
 * map, normal/depth pass, scene, edges, bloom, SMAA) costs ~4.6 ms at device pixel ratio 1,
 * ~8.5 ms at 1.5 and ~14 ms at 2. Drawn on every refresh of a 144 Hz display that is 0.65-2 s of
 * GPU work per second, idle or not. So:
 * - at most MAX_FPS frames a second, whatever the display's refresh rate;
 * - a frame where nothing changed (paused, camera still) is not drawn, except IDLE_FPS times a
 *   second so the park's flags, waterfall and clock keep moving;
 * - device pixel ratio capped at MAX_DPR.
 */
export const MAX_FPS = 60;
export const IDLE_FPS = 20;
// 1.5 -> 2 is 1.78x the pixels and ~1.65x the frame time (8.5 -> 14 ms measured); at 1.5 the ink
// lines and SMAA already keep edges clean, so the extra sharpness is not worth it.
export const MAX_DPR = 1.5;
// A frame this early still counts as on time: display frames jitter by a ms or so.
const SLACK_MS = 2;

/**
 * Frame cap: returns `due(now)` (now = requestAnimationFrame time, ms), true on every n-th display
 * refresh, n = round(refresh rate / fps) (at least 1): 60 Hz and 120 Hz -> 60 fps, 144 Hz -> 72, 165 -> 55,
 * 240 -> 60. Why every n-th refresh, not a fixed 1/60 s schedule: a 144 Hz display cannot show frames
 * 16.7 ms apart, so that schedule held them for 3, 2, 2, 3... refreshes (20.8 / 13.9 ms gaps). Each
 * frame still showed the right instant, but the uneven hold reads as judder, worst in slow motion
 * (1/4x, 1/10x) where the eye tracks the pitcher's slow, smooth movement. Even gaps fix it.
 * The refresh interval is the median of the last 60 rAF gaps (robust to dropped frames and stalls).
 */
export function makeFrameCap(fps = MAX_FPS) {
  const gaps = [];
  let last = null, k = 0;
  return (now) => {
    if (last !== null) { gaps.push(now - last); if (gaps.length > 60) gaps.shift(); }
    last = now;
    const refreshMs = gaps.length ? [...gaps].sort((a, b) => a - b)[gaps.length >> 1] : 1000 / fps;
    return k++ % Math.max(1, Math.round(1000 / fps / refreshMs)) === 0;
  };
}

/** Draw this (due) frame? Yes if anything changed, else only at the idle rate. Times in ms. */
export function shouldDraw(changed, now, lastDraw, idleFps = IDLE_FPS) {
  return changed || now - lastDraw >= 1000 / idleFps - SLACK_MS;
}

/** Renderer pixel ratio for a display's devicePixelRatio; the lighter-graphics mode draws at 1. */
export const pixelRatio = (dpr, lite = false) => (lite ? 1 : Math.min(dpr, MAX_DPR));
