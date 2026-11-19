import "./cards.css"; // not ui.css: that one styles the viewer's full-screen body
import { boot } from "./boot.js";

boot(() => import("./compare.js"));
