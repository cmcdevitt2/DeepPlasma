import { defineConfig } from "vite";

export default defineConfig({
  // Relative base makes the built demo work under GitHub Pages subpaths.
  base: "/DeepPlasma/Demo_LDC/demo/",
  build: {
    outDir: "dist",
  },
});
