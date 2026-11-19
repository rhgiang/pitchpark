// FROZEN: a verbatim copy of web/src/{pitcher,figure,skeleton}.js from commit 70c9f22, the viewer
// Ryan signed off for Ohtani's delivery (gaze yaw 85 deg, refined figure, height-scaled rig).
// Do not edit. Shared viewer changes must not change Ohtani unless Ryan asks.
// Ohtani's JSON (results/m5/ohtani-shohei/signed-off/) pairs with this code; later fixes
// (per-pitcher bone lengths, foot/rocker fixes, arm blend) changed his stride and are not applied.
import * as current from "../pitcher.js";
import * as frozen from "./pitcher.js";

// ?legfix=0 turns Ohtani's leg-jitter fix off (A/B by eye); see frozen.legfix in pitcher.js.
frozen.legfix.on = new URLSearchParams(globalThis.location?.search).get("legfix") !== "0";

export const SIGNED_OFF = new Set(["ohtani-shohei"]);

/** The pose API (makePitcher, poseAtTime, poseAtRelease, handPosition) for this pitcher. */
export const pitcherApi = (slug) => (SIGNED_OFF.has(slug) ? frozen : current);
