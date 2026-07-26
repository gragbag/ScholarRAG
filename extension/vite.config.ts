import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { crx } from "@crxjs/vite-plugin";
import manifest from "./manifest";

// CRXJS wires the MV3 manifest into Vite: it builds the popup, background worker,
// and content script with the right formats, and rewrites the manifest paths.
export default defineConfig({
  plugins: [react(), crx({ manifest })],
  // A stable port so the extension's dev server / HMR is predictable.
  server: { port: 5173, strictPort: true, hmr: { port: 5173 } },
});
