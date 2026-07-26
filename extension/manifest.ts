import { defineManifest } from "@crxjs/vite-plugin";

// The Manifest V3 manifest — the extension's "identity card". It declares what
// the extension IS (name, popup, scripts) and, crucially, what it's ALLOWED to
// do (permissions, host_permissions). Chrome enforces every line here.
export default defineManifest({
  manifest_version: 3,
  name: "ScholarRAG",
  version: "0.1.0",
  description: "Save the page you're reading into a folder, then ask questions over it.",

  // The toolbar-icon popup — our React UI.
  action: { default_popup: "index.html", default_title: "ScholarRAG" },

  // The background service worker (event handling; runs without a page).
  background: { service_worker: "src/background.ts", type: "module" },

  // The content script — injected into every page so it can read the article
  // text with Readability when the popup asks.
  content_scripts: [
    {
      matches: ["<all_urls>"],
      js: ["src/content.ts"],
      run_at: "document_idle",
    },
  ],

  permissions: ["identity", "storage", "activeTab", "scripting"],
  host_permissions: ["http://localhost:8001/*"],
});
