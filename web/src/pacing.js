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
 * Frame cap. `next` is when the next frame is due (ms, requestAnimationFrame time).
 * Keeps a fixed 1/MAX_FPS schedule, so a 144 Hz display averages 60 fps (not 48, as "skip until
 * 16.7 ms have passed" would give); after a stall it resyncs rather than bursting.
 * @returns {{due: boolean, next: number}}
 */
export function frameSlot(now, next, fps = MAX_FPS) {
  if (now < next - SLACK_MS) return { due: false, next };
  const step = 1000 / fps;
  const n = next + step;
  return { due: true, next: n < now ? now + step : n };
}

/** Draw this (due) frame? Yes if anything changed, else only at the idle rate. Times in ms. */
export function shouldDraw(changed, now, lastDraw, idleFps = IDLE_FPS) {
  return changed || now - lastDraw >= 1000 / idleFps - SLACK_MS;
}

/** Renderer pixel ratio for a display's devicePixelRatio. */
export const pixelRatio = (dpr) => Math.min(dpr, MAX_DPR);
