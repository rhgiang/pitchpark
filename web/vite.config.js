import { defineConfig } from "vite";
import { resolve } from "node:path";

export default defineConfig({
  // three.js alone is ~540 kB minified; one chunk is fine for a local viewer.
  // Three pages: the viewer, the M5 eye-test page (compare.html) and 404.html (Vercel serves dist/404.html for unknown paths).
  build: {
    chunkSizeWarningLimit: 1000,
    rollupOptions: { input: { main: resolve(import.meta.dirname, "index.html"), compare: resolve(import.meta.dirname, "compare.html"), notFound: resolve(import.meta.dirname, "404.html") } },
  },
});
