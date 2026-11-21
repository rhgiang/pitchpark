# Load performance (web viewer)

Goal: the site loads fast. Site code was already fine (230 KB gzipped) and the deliveries are small
(0.9 MB total); the weight was the per-pitcher pitch data, 5-15 MB each.

## What was done

1. **Slimmer pitch data (export schema 2 -> 3).** Every pitch used to store three paths (`real`, `sim`,
   `nospin`) as 5 ms samples. Now:
   - `real` is the Statcast 9-parameter fit itself (`f` = pos, vel, acc at y = 50 ft; `r` = release time
     on the fit's clock; `T` = flight time). The viewer evaluates it analytically.
   - `sim` and `nospin` are RK4 knots every 50 ms (position + velocity) plus one at the plate; the viewer
     joins them with a cubic Hermite spline.
   - `check_real` / `check_nospin` (truth points for the tests) are only written for the test fixture
     (`export_pitcher(checks=True)`), not shipped.
   - Plate times are stored to 1e-6 s (they were 1e-4), positions to 1e-4 ft, knot velocities to 1e-3 ft/s.
   - `web/src/trajectory.js` `expandPitcher` turns a schema-3 file back into the `{t, p}` 5 ms paths the
     ball, trail, ghost, ride camera, HUD and tunnel code already read (expanded lazily per pitch, on first
     access), so none of those changed. Any other schema throws, which `loadOrAsk` shows as the data-error card.
   - `index.json` rows gained `height_in` and `mix` (pitch counts by type), so the start menu no longer
     downloads every pitcher's file in the background just to draw its cards. Same pitchers, same order.
2. **Prefetch.** Hovering, focusing or selecting a start-menu card starts that pitcher's data + delivery
   download (`data.js` `prefetchPitcher`). Requests are cached per URL (in-flight included), so Play ball
   reuses the same promise; a failed request is dropped from the cache so Retry really retries.
3. **Caching.** `web/vercel.json` (Vercel's root directory is `web/`): `/assets/*` (hashed) `immutable`, one
   year; `/data/*` `max-age=3600, stale-while-revalidate=86400`.
4. **Shader warm-up.** `pickPitcher` ends with `renderer.compileAsync(scene, camera)` while the loading chip
   is up. three's compile skips invisible objects, so the ball, ghost and trails are shown for the call and
   restored. Capped at 3 s; a failure is ignored. Not measured: first-frame time needs a real GPU browser
   (headless has no useful GPU), so check by eye that the first pitch no longer hitches.

## Sizes (all 8 pitchers re-exported)

| pitcher | before raw | before gzip | after raw | after gzip |
|---|---|---|---|---|
| cease-dylan | 15.29 MB | 4.84 MB | 3.10 MB | 0.91 MB |
| king-michael | 7.20 MB | 2.31 MB | 1.45 MB | 0.44 MB |
| misiorowski-jacob | 8.63 MB | 2.71 MB | 1.83 MB | 0.53 MB |
| ohtani-shohei | 5.66 MB | 1.80 MB | 1.16 MB | 0.34 MB |
| sale-chris | 6.37 MB | 2.05 MB | 1.32 MB | 0.39 MB |
| sanchez-cristopher | 7.12 MB | 2.30 MB | 1.47 MB | 0.43 MB |
| schlittler-cam | 10.03 MB | 3.15 MB | 2.08 MB | 0.60 MB |
| yamamoto-yoshinobu | 13.06 MB | 4.15 MB | 2.66 MB | 0.79 MB |
| **total** | **73.36 MB** | **23.32 MB** | **15.07 MB** | **4.44 MB** |

One pitcher is now 0.35-0.9 MB gzipped (was 1.8-4.8 MB).

## Accuracy: new format against the old, every pitch

`node web/tools/compare-exports.mjs <old dir> <new dir>` expands the new files and compares them with the
old 5 ms samples at every old sample time (8 pitchers, 8,867 pitches, all three paths).
Old files were the previous exports of the same CSVs (the pitcher list and order in `index.json` are identical).

| pitcher | pitches | real (ft) | sim (ft) | nospin (ft) | plate position (ft) |
|---|---|---|---|---|---|
| cease-dylan | 1811 | 0.00017 | 0.00014 | 0.00014 | 0.00000 |
| king-michael | 831 | 0.00016 | 0.00014 | 0.00014 | 0.00000 |
| misiorowski-jacob | 1105 | 0.00017 | 0.00014 | 0.00015 | 0.00000 |
| ohtani-shohei | 680 | 0.00017 | 0.00015 | 0.00015 | 0.00000 |
| sale-chris | 777 | 0.00016 | 0.00014 | 0.00015 | 0.00000 |
| sanchez-cristopher | 856 | 0.00017 | 0.00014 | 0.00014 | 0.00000 |
| schlittler-cam | 1255 | 0.00016 | 0.00015 | 0.00015 | 0.00000 |
| yamamoto-yoshinobu | 1552 | 0.00016 | 0.00014 | 0.00014 | 0.00000 |

Maximum over everything: **real 0.00017 ft, sim 0.00015 ft, no-spin 0.00015 ft** (0.002 in), against a
0.01 ft requirement; plate position identical (to 1e-4 ft). Plate time differs by 5e-5 s at most, and that
is the old file's own rounding (it stored the plate time to 1e-4 s and evaluated the position at the
unrounded time; the new file stores 1e-6 s). The last sample is compared point to point for that reason;
comparing at the old rounded time would show 0.0065 ft that belongs to the old file.
The error is mostly the old data's own 1 ms-to-5 ms chord interpolation and the 4-decimal rounding of
the fit; the Hermite error at 50 ms is far below it (about h^4/384 times the fourth derivative of position).

Tests (`web/test/trajectory.test.js`): Hermite reproduces a cubic exactly, the fit grid ends exactly at T,
an unknown schema throws, and the existing truth tests (fixture: RK4 check points, 0.002 in) pass on the new
format; `web/test/data.test.js`: one download per file, failures not cached, old schema rejected.

## Left alone / not measured

- Plain 5 ms samples remain in memory after expansion (about 100 points per path, lazily): load time is
  download-bound, not CPU-bound.
- Real-GPU first-frame time before/after the warm-up was not measured.
- `web/public/data/` is gitignored; the deployment needs the re-exported files as well.

## Frame time: Sim vs Replay (M7, 2026-11-20)

Method as in `results/M4.md` "Frame rate": the `?fps=1` burst (100 frames back to back, then a GPU sync),
production build (`vite build`, `vite preview`), headless Edge at 1600x900, umpire view, day, Chris Sale,
frozen mid-flight (`t=0.4`). Replay: `?pitcher=sale-chris&view=umpire&ab=0&pitch=2`. Sim:
`?face=1&stand=R&batter=avg&pitcher=sale-chris&view=umpire&pitch=2`. Three runs each, alternating.

| Mode | ms/frame (3 runs) | with shadow redraw | draws | triangles |
|---|---|---|---|---|
| Replay, umpire | 5.69, 5.36, 5.46 | 6.3-8.5 | 365 | 3.72 M |
| Sim, umpire | 5.46, 5.21, 5.14 | 8.3-8.4 | 365 | 3.72 M |

Sim costs the same as Replay (it is the same scene and the same ball path; both are GPU-bound and the
difference is inside the +-0.4 ms run-to-run noise). Both are far under 16.7 ms. These are higher than the
M4 figures (2.5 ms) because the park has grown since (crowd, art passes). The shadow-redraw frame is the one
after a pitch change; Sim's is a little higher in these three runs (8.3-8.4 vs 6.3-8.5), which is within burst noise.

Hardware caveat: this machine's GPU through headless Edge (M4 recorded an RTX 5060 Ti here; the GPU was not queried again). A software renderer
(SwiftShader) gives about 1,200 ms/frame, so it says nothing about a laptop or a phone. Burst timing says
nothing about live rAF pacing; open either mode with `?fps=1` on the target device to see live fps.
