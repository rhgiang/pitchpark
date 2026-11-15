import { defineConfig } from "vite";
import { resolve } from "node:path";

export default defineConfig({
  // three.js alone is ~540 kB minified; one chunk is fine for a local viewer.
  // Two pages: the viewer and the M5 eye-test page (compare.html).
  build: {
    chunkSizeWarningLimit: 1000,
    rollupOptions: { input: { main: resolve(import.meta.dirname, "index.html"), compare: resolve(import.meta.dirname, "compare.html") } },
  },
});
