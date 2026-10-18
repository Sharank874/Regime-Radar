import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// `--mode static` builds a server-less copy for GitHub Pages that reads
// pre-exported JSON from /static-data (see `python -m radar export-static`).
export default defineConfig(({ mode }) => ({
  plugins: [react()],
  base: mode === "static" ? "./" : "/",
  publicDir: mode === "static" ? "static-public" : "public",
  define: { __STATIC__: JSON.stringify(mode === "static") },
  server: { proxy: { "/api": "http://127.0.0.1:8000", "/ws": { target: "ws://127.0.0.1:8000", ws: true } } },
}));
