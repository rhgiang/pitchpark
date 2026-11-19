// Dev-only test switches: ?debug=nowebgl,indexfail,slow,contextloss,contextloss-stuck (comma list).
// import.meta.env.DEV is false in a production build, so none of this exists on the hosted site.
import { debugList } from "./health.js";

const on = import.meta.env?.DEV ? debugList(globalThis.location?.search ?? "") : [];
export const debugHas = (name) => on.includes(name);
