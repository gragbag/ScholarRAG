# ScholarRAG Browser Extension

A Manifest V3 (Vite + React + TypeScript) extension: sign in with Google, save
the page you're reading into a folder, and ask questions scoped to that folder —
all against your ScholarRAG backend.

```
manifest.ts     the extension's declared identity + permissions   (exercise 1)
src/auth.ts     Google sign-in via chrome.identity                 (exercise 2)
src/page.ts     read the current tab's text (→ content.ts)         (exercise 3)
src/content.ts  Readability extraction, injected into every page
src/api.ts      backend client (/auth/google, /folders, /documents/text, /query)
src/Popup.tsx   the React popup UI  ← redesign this freely
```

## First-time setup

### 1. Install + build

```bash
cd extension
npm install
npm run build        # produces dist/  (or: npm run dev for hot-reload)
```

### 2. Load it in Chrome

`chrome://extensions` → toggle **Developer mode** (top-right) → **Load unpacked**
→ pick `extension/dist`. Note the **extension ID** Chrome assigns — you need it next.

### 3. Create the Google OAuth client

Google Cloud Console → **APIs & Services → Credentials → Create credentials →
OAuth client ID** → type **Web application**. Under **Authorized redirect URIs**
add exactly:

```
https://<YOUR_EXTENSION_ID>.chromiumapp.org/
```

Copy the **Client ID** into two places:
- `src/config.ts` → `GOOGLE_CLIENT_ID`
- your backend `.env` → `GOOGLE_CLIENT_ID=...` (same value), then restart `make run`

### 4. Run the backend

`make run` (API on :8001). The extension's `API_BASE` already points there.

Rebuild the extension (`npm run build`) and hit the reload ↻ on the extensions
page whenever you change files (or use `npm run dev`).

---

## Exercises

### Exercise 1 — permissions (`manifest.ts`)
Fill `permissions` (`identity`, `storage`, `activeTab`, `scripting`) and
`host_permissions` (`http://localhost:8001/*`). Without these, Chrome blocks the
OAuth call, storage, and the fetch to your backend. **Verify:** the extension
loads with no manifest error and the popup opens.

### Exercise 2 — Google sign-in (`src/auth.ts` → `signIn`)
Implement the `launchWebAuthFlow` → ID token → `/auth/google` → store-JWT flow
(guided in the file). **Verify:** click "Sign in with Google," approve, and the
popup switches to the signed-in view (folder + ask boxes).

### Exercise 3 — read the current page (`src/page.ts` → `extractActivePage`)
Query the active tab and message the content script for its Readability text.
**Verify:** open an article, click **Add page**, then ask a question about it and
get a grounded answer.

Do them in order — each unlocks the next. When all three are done, the whole loop
(sign in → add page → ask) works end to end.
