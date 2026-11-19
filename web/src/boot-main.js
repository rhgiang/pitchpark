import "./ui.css"; // the cards need it before main.js (and its own import) has loaded
import { boot } from "./boot.js";

boot(() => import("./main.js"));
