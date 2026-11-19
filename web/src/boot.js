// Entry for both pages: check for WebGL and install the last-resort error cards BEFORE the page's own
// module runs (it creates a renderer on its first lines). `load` is e.g. `() => import("./main.js")`.
import { canMakeWebGL } from "./health.js";
import { showNoWebGL, installGlobalHandlers } from "./notices.js";
import { debugHas } from "./debug.js";

export async function boot(load) {
  installGlobalHandlers();
  if (debugHas("nowebgl") || !canMakeWebGL()) return showNoWebGL();
  try {
    await load();
  } catch (err) {
    console.error("page failed to start:", err);
    if (/webgl|context/i.test(String(err?.message))) showNoWebGL(); // the renderer constructor threw
    else throw err; // anything else: the global handler shows the generic card
  }
}
