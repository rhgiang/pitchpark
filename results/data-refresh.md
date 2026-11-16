# Data refresh: second seasons for Ohtani and Misiorowski, plus Dylan Cease (2024 + 2026)

All numbers are on held-out games. Run with `scripts/validate.py`, `train_generative.py` and
`train_seam.py`, passing each pitcher's season CSVs together, with and without `--cross-season`.
Then `m4_checks.py` and `export_viewer.py` over all six pitchers. The pre-refresh numbers are copied
in `results/before-refresh/` (gitignored, like all result tables).

## What was added

| Pitcher | Seasons | Games | Pitches |
|---|---|---|---|
| Ohtani | 2025 (had) + **2026** | 14 + 14 | 753 + 1,335 |
| Misiorowski | **2025** + 2026 (had) | 15 + 30 | 1,165 + 2,894 |
| Cease (new) | **2024** + **2026** | 33 + 29 | 3,188 + 2,896 |

Sale, King and Sánchez are unchanged (2026 only).

## Season-mixing rule

- **Pooled (the main numbers):** a pitcher's seasons are joined and whole games are held out at random,
  as before (`split_by_game`, seed 0). This measures whether more data helps.
- **Cross-season (`--cross-season`):** train on the older season, test on the newer one
  (`data.mark_cross_season`). This measures how well a model survives a year of drift.
- **One relabel** (`data.RELABEL`): Statcast's "SL" for Misiorowski in 2025 (283 pitches) is
  modelled as **FC**. By season means it is his 2026 cutter: 94.1 vs 95.8 mph, axis 181 vs 195°,
  6.4 vs 7.2 in of rise. It is not his 2026 slider (91.9 mph, 124°, 1.8 in). Without the relabel,
  one "SL" model would mix two pitches.

## M1: physics (median held-out plate miss, inches, release start + per-game air)

| Pitcher | Type | Before | Pooled | Cross-season |
|---|---|---|---|---|
| Ohtani | FF | 4.02 (92 test) | 4.33 (324) | 4.53 (602) |
| Ohtani | ST | 9.14 (27) | 8.23 (190) | 8.37 (393) |
| Ohtani | FS | n/a | 7.67 (42) | 7.35 (114) |
| Misiorowski | FF | 2.03 | 2.02 | 2.11 |
| Misiorowski | FC | 4.54 | 3.39 | 3.66 |
| Misiorowski | SL | 4.44 | 3.26 | (2026 only, n/a) |
| Misiorowski | CU | 3.84 | 4.19 | 3.56 |
| Cease | FF | | 2.36 | 2.31 |
| Cease | SL | | 2.07 | 2.25 |
| Cease | KC | | 4.27 | 5.18 |
| Cease | ST | | 4.12 | 4.94 |
| Cease | CH | | 5.17 | n/a (28 in 2024) |
| Cease | SI | | 8.37 | n/a (2 in 2024) |

- **The physics carries across seasons.** Cd and spin efficiency fitted on one season miss the next by
  about as much as they miss held-out games of the same season (within ~0.9 in everywhere).
- Ohtani's earlier numbers rested on 5–27 test pitches per type. The pooled ones are the first
  that are trustworthy. His sweeper (8.2 in) and splitter (7.7 in) are still the worst M1 fits, and
  M3 exists for exactly those.

## M2: the arm (classifier two-sample test; AUC 0.5 = can't tell real from generated)

Pooled results (since replaced, see the decision below) were in the usual range: AUC 0.57–0.79, with a floor (real vs real) of 0.56–0.74.
Pass counts are in the summary CSVs. The pooled results lose a couple of passes against single
seasons, e.g. Ohtani FF 0.70 vs floor 0.68 now fails, because one Gaussian now has to cover two
seasons' release points.

**Cross-season is the real finding: the floor itself is 0.94–0.99 for every graded type.** A classifier can tell one season's
*real* pitches from the next season's *real* pitches almost perfectly. So the cross-season "passes" mean
nothing: the model matches its training season, and the test season is a different distribution.
The main reason is that the **release point moves sideways** between seasons, by 1–2 times its
within-season spread for fastballs. Here is (newer − older) / within-season sd:

| FF | release x | release z | extension | velo | spin | axis |
|---|---|---|---|---|---|---|
| Cease 24→26 | −1.75 | 0.09 | 0.87 | −0.04 | −0.95 | 0.81 |
| Ohtani 25→26 | 1.04 | −0.66 | −0.24 | −0.24 | 0.20 | 0.04 |
| Misiorowski 25→26 | −1.98 | −0.32 | −0.09 | 0.85 | 0.59 | 0.87 |

Whether that is a real change (rubber position, arm slot) or a change in Hawk-Eye calibration between
seasons can't be told from this data.

**Decision (Ryan): M2 is fitted on the latest season only.** `train_generative.py` keeps the newest
`game_year` unless `--cross-season` is given, and the pooled run above was replaced. The result is
graded types passing with the best of Gaussian or mixture, held out within the latest season:

| Pitcher | Pooled (both seasons) | Latest season only |
|---|---|---|
| Cease (2026) | 5 of 6 (CH fails) | **6 of 6** |
| Ohtani (2026) | 3 of 5 (FF, SI fail) | 3 of 4 (CU fails; SI ungraded, 25 test pitches) |
| Misiorowski (2026) | 3 of 4 (SL fails) | 3 of 4 (SL fails; identical to the pre-refresh 2026 run) |

M1 and M3 keep pooling seasons. Their fits carry across years, and more data helps them.

## M3: extra push (median miss M1 only → M1 + per-pitch push; pass = drop ≥ 0.6 in and interval above 0)

| Pitcher | Type | Before | Pooled | Cross-season |
|---|---|---|---|---|
| Ohtani | FF | 4.02 → 2.02 pass | 4.33 → 2.19 pass | 4.53 → 2.28 pass |
| Ohtani | ST | ungraded (27) | 8.23 → 2.93 **pass** | 8.37 → 3.27 pass |
| Ohtani | SI | ungraded (5) | 8.39 → 3.06 **pass** | 8.18 → 3.24 pass |
| Ohtani | FS | n/a | 7.67 → 3.44 **pass** | 7.35 → 3.37 pass |
| Ohtani | CU | ungraded (7) | 2.05 → 1.72 fail | 2.58 → 2.60 fail |
| Misiorowski | FC | 3.93 → 1.87 pass | 3.39 → 1.88 pass | 3.66 → 1.78 pass |
| Misiorowski | SL | 2.86 → 2.18 fail | 3.26 → 1.94 **pass** | n/a |
| Misiorowski | CU | 3.84 → 3.14 pass | 4.19 → 2.86 pass | 3.56 → 3.03 fail |
| Misiorowski | FF | 2.03 → 1.64 fail | 2.02 → 1.54 fail | 2.11 → 1.77 fail |
| Cease | SI | | 8.37 → 2.16 pass | n/a |
| Cease | CH | | 5.17 → 1.80 pass | n/a |
| Cease | KC | | 4.27 → 2.18 pass | 5.18 → 3.09 pass |
| Cease | ST | | 4.12 → 2.85 pass | 4.94 → 3.59 pass |
| Cease | FF | | 2.36 → 1.84 fail | 2.31 → 1.76 fail |
| Cease | SL | | 2.07 → 1.83 fail | 2.25 → 1.85 fail |

- **More data turned four ungraded or failing types into passes** (Ohtani ST, SI, FS; Misiorowski SL).
  Ohtani went from 1 graded type to 5.
- **The push carries across seasons.** Where a type passes pooled, it also passes cross-season, except
  Misiorowski's CU (drop 0.54, just under 0.6). The extra movement is a stable habit of the pitch, not
  noise from one season.
- The fails are pitches M1 already fits well (2.0–2.4 in), so there is little left for a push to take.

## Dylan Cease

- **The slider is a gyro slider.** M1 fits its spin efficiency at **0.05**: almost all of its 2,700 rpm
  is bullet spin, which doesn't move the ball. That explains three things at once:
  - its small movement (about 2 in each way);
  - its measured spin axis, which scatters ±30° and whose average moved 35° from 2024 to 2026
    (98° → 133°) while the movement barely changed;
  - why M1 alone already fits it well (2.07 in) and M3 adds little (0.24 in drop, fail). With so
    little useful spin, the axis hardly matters.
- **The arsenal changed from 2024 to 2026.** Sliders went from 43% to 30%. The changeup went from
  28 pitches to 332 and the sinker from 2 to 228. Both new pitches carry big seam effects: the sinker
  runs **8.3 in** more to the arm side than its spin explains, and the changeup **5.1 in**. M3 cuts
  those misses to 2.2 and 1.8 in.
- His knuckle curve and sweeper keep the same push across the two-year gap (both pass cross-season).

## M4 checks and viewer

- Perceived velocity vs Statcast `effective_speed`, held out, pooled over all six pitchers:
  median |diff| **0.235 mph** (was 0.251, now on 6,028 pitches instead of 3,400). The best reference
  extension on the training games is 6.33 ft against the 6.34 in use, so no retune.
- The viewer now has Cease (422 held-out at-bats; 2 pitches are `no_model`, a type too rare to fit, so the viewer
  shows them as the Statcast path only) and both seasons for Ohtani and Misiorowski.

## Still wrong / open

- M2 now ignores older seasons. "Season as context" would use them, if the latest season is ever too thin.
- The Misiorowski relabel is a judgement from season means, not from Statcast. If Savant's classifier
  changes again, `RELABEL` needs another look.
- Cease's 2024 changeup and sinker are too rare to test cross-season.

---

# Second refresh: Cam Schlittler and Yoshinobu Yamamoto (2025 + 2026)

Same routine as above, one run per pitcher with both seasons: `validate.py` and `train_seam.py` pooled
(held-out games, seed 0), `train_generative.py` on the latest season only, then `m4_checks.py` and
`export_viewer.py` over all eight pitchers. Cross-season runs were not done for these two.

| Pitcher | Season | Games | Pitches |
|---|---|---|---|
| Schlittler | 2025 / 2026 | 14 / 33 | 1,253 / 3,044 |
| Yamamoto | 2025 / 2026 | 30 / 28 | 2,792 / 2,740 |

## M1 and M3 (pooled, median held-out plate miss in inches; M3 pass = drop >= 0.6 in with interval above 0)

| Pitcher | Type | n test | M1 | M1 + push | M3 |
|---|---|---|---|---|---|
| Schlittler | FF | 608 | 2.55 | 1.72 | pass |
| Schlittler | FC | 332 | 4.42 | 1.95 | pass |
| Schlittler | SI | 166 | 7.56 | 2.06 | pass |
| Schlittler | CU | 119 | 4.10 | 2.78 | pass |
| Schlittler | SL | 12 | 4.33 | 4.58 | ungraded (12 test pitches) |
| Yamamoto | FF | 460 | 2.92 | 1.81 | pass |
| Yamamoto | FS | 411 | 7.00 | 3.33 | pass |
| Yamamoto | SI | 173 | 6.15 | 2.15 | pass |
| Yamamoto | FC | 205 | 4.05 | 2.22 | pass |
| Yamamoto | CU | 208 | 1.89 | 1.92 | fail (M1 already good) |
| Yamamoto | SL | 89 | 2.79 | 2.61 | fail (M1 already good) |

Both pitchers' sinkers (and Yamamoto's splitter) show the usual large seam push that M1 alone can't
explain (7.6, 6.2 and 7.0 in), which M3 mostly removes.

## M2 (latest season, AUC of real vs generated; floor = real vs real; pass = within the floor's interval)

| Pitcher | Type | AUC (best model) | Floor | Pass |
|---|---|---|---|---|
| Schlittler | FF | 0.67 (Gaussian) | 0.71 | yes |
| Schlittler | FC | 0.70 | 0.70 | yes |
| Schlittler | SI | 0.66 | 0.64 | yes |
| Schlittler | CU | 0.59 | 0.60 | yes |
| Yamamoto | FF | 0.63 | 0.59 | no |
| Yamamoto | FC | 0.62 | 0.57 | no |
| Yamamoto | FS | 0.64 | 0.64 | yes |
| Yamamoto | SI | 0.56 | 0.55 | yes |
| Yamamoto | CU | 0.61 | 0.61 | yes |
| Yamamoto | SL | 0.59 | 0.51 | no (123 train pitches) |

Schlittler 4 of 4 graded types pass (SL ungraded, under 30 training pitches). Yamamoto 3 of 6. His fails are
small gaps above the floor (0.03 to 0.08), and the mixture never beat the single Gaussian for either pitcher.

## M4 and viewer

- Perceived velocity vs `effective_speed`, held out, all eight pitchers (13 CSVs): median |diff| **0.244 mph**
  (0.235 before). Best reference extension 6.31 ft vs 6.34 in use, so still no retune.
- Viewer now lists eight pitchers; new: Schlittler (310 at-bats, 17 pitches `no_model` from SL/ST too rare to fit)
  and Yamamoto (401 at-bats).

## Still wrong

- Schlittler's 2025 is only 14 games, and his slider is 43 pitches pooled, so it stays ungraded.
- Heights for the viewer come from StatsAPI via `people.csv`; M2 for Yamamoto fails on 3 of 6 types and has
  not been looked into beyond the AUC table.
