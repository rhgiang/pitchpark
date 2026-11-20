import { defineConfig } from "vite";
import { resolve } from "node:path";

export default defineConfig({
  // three.js alone is ~540 kB minified; one chunk is fine for a local viewer.
  // Two built pages: the viewer and 404.html (Vercel serves dist/404.html for unknown paths). compare.html (the M5
  // eye-test page) is dev-only: it needs Ryan's own clips, and its clip data is not shipped (see .gitignore).
  build: {
    chunkSizeWarningLimit: 1000,
    rollupOptions: { input: { main: resolve(import.meta.dirname, "index.html"), notFound: resolve(import.meta.dirname, "404.html") } },
  },
});
