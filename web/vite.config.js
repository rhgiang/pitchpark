import { defineConfig } from "vite";

export default defineConfig({
  // three.js alone is ~540 kB minified; one chunk is fine for a local viewer.
  build: { chunkSizeWarningLimit: 1000 },
});
