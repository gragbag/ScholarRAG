// Background service worker. Minimal here — sign-in runs in the popup via
// chrome.identity, and extraction runs in the content script. It exists because
// MV3 requires a declared worker, and it's where you'd add context menus,
// alarms, or a right-click "Add to ScholarRAG" later.

chrome.runtime.onInstalled.addListener(() => {
  console.log("ScholarRAG extension installed");
});
