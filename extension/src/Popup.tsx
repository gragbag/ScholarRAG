// The popup UI. It's wired to the API + auth + page-extraction helpers — but the
// LOOK is yours: restyle it, rearrange it, make it your own. It only works once
// you've done the three exercises (manifest permissions, signIn, extractActivePage).

import { useEffect, useState } from "react";
import {
  addPageText,
  addPageUrl,
  createFolder,
  deleteDocument,
  listDocuments,
  listFolders,
  listFolderSummaries,
  query,
  renameDocument,
  type Answer,
  type DocItem,
} from "./api";
import { signIn, signOut } from "./auth";
import { extractActivePage } from "./page";
import { getFolder, getToken, setFolder } from "./storage";

// A folder + its document count. count === null means "unknown" — we're in the
// fallback path because the backend's folder_summaries (your exercise) isn't
// implemented yet, so we only have names from GET /folders.
type Folder = { name: string; count: number | null };

export function Popup() {
  const [signedIn, setSignedIn] = useState(false);
  const [folders, setFolders] = useState<Folder[]>([]);
  const [folder, setFolderState] = useState("default");
  const [creating, setCreating] = useState(false);
  const [newFolder, setNewFolder] = useState("");
  const [docs, setDocs] = useState<DocItem[]>([]);
  const [pageTitle, setPageTitle] = useState(""); // editable name for the page being added
  const [editingId, setEditingId] = useState<string | null>(null); // doc being renamed
  const [editName, setEditName] = useState("");
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState<Answer | null>(null);
  const [status, setStatus] = useState("");

  // Load folders with counts; fall back to names-only if the count endpoint 500s
  // (i.e. folder_summaries isn't implemented yet). Always include the active
  // folder so a freshly-created / empty folder stays visible and selectable.
  async function loadFolders(active: string): Promise<Folder[]> {
    let list: Folder[];
    try {
      list = (await listFolderSummaries()).map((f) => ({ name: f.name, count: f.count }));
    } catch {
      list = (await listFolders()).map((name) => ({ name, count: null }));
    }
    if (active && !list.some((f) => f.name === active)) {
      list = [...list, { name: active, count: 0 }];
    }
    return list;
  }

  async function refresh() {
    const token = await getToken();
    setSignedIn(!!token);
    const active = await getFolder();
    setFolderState(active);
    if (token) {
      try {
        setFolders(await loadFolders(active));
      } catch {
        /* not signed in yet, or backend down */
      }
    }
  }

  useEffect(() => {
    void refresh();
    // Pre-fill the name box with the current tab's title (editable before adding).
    void chrome.tabs.query({ active: true, currentWindow: true }).then(([tab]) => {
      if (tab?.title) setPageTitle(tab.title);
    });
  }, []);

  // Load the pages in the active folder whenever it changes (or on sign-in).
  useEffect(() => {
    if (!signedIn) return;
    void listDocuments(folder)
      .then(setDocs)
      .catch(() => setDocs([]));
  }, [folder, signedIn]);

  async function run(label: string, fn: () => Promise<void>) {
    setStatus(label);
    try {
      await fn();
      setStatus("");
    } catch (e) {
      setStatus(String(e instanceof Error ? e.message : e));
    }
  }

  const handleSignIn = () =>
    run("Signing in…", async () => {
      await signIn();
      await refresh();
    });

  const onFolderChange = (name: string) => {
    setFolderState(name);
    void setFolder(name);
  };

  const commitNewFolder = () => {
    const name = newFolder.trim();
    setCreating(false);
    setNewFolder("");
    if (!name) return;
    onFolderChange(name);
    void run(`Creating "${name}"…`, async () => {
      try {
        await createFolder(name); // persist it server-side
        setFolders(await loadFolders(name));
      } catch {
        // create_folder (backend exercise) not implemented yet → optimistic add
        // so the chip still shows this session (it just won't survive a reload).
        setFolders((prev) =>
          prev.some((f) => f.name === name) ? prev : [...prev, { name, count: 0 }],
        );
      }
    });
  };

  const handleAddPage = () =>
    run("Adding this page…", async () => {
      const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

      // Try client-side Readability first. On a PDF page (arXiv, etc.) there is
      // no article DOM — extraction comes back empty, or the content script was
      // never injected into the PDF viewer — so we fall back to letting the
      // server fetch the URL (it sniffs application/pdf → runs the PDF pipeline).
      let extracted = tab?.title ?? "page";
      let text = "";
      try {
        const page = await extractActivePage();
        extracted = page.title || extracted;
        text = page.text;
      } catch {
        // no content script on this page (PDF viewer / chrome:// / store page)
      }

      const name = pageTitle.trim() || extracted; // the user's edited name wins

      if (text.trim().length > 50) {
        await addPageText(text, name, folder);
      } else if (tab?.url) {
        await addPageUrl(tab.url, folder, name); // PDF / unextractable → server fetches
      } else {
        throw new Error("Nothing to add — couldn't read this page.");
      }

      setStatus(`Added "${name}" to ${folder}`);
      setFolders(await loadFolders(folder));
      setDocs(await listDocuments(folder).catch(() => docs));
    });

  const handleDelete = (id: string) =>
    run("Removing…", async () => {
      await deleteDocument(id);
      setDocs(await listDocuments(folder).catch(() => []));
      setFolders(await loadFolders(folder)); // count drops
    });

  const startRename = (d: DocItem) => {
    setEditingId(d.id);
    setEditName(d.filename);
  };

  const commitRename = (id: string) => {
    const title = editName.trim();
    setEditingId(null);
    if (!title) return;
    void run("Renaming…", async () => {
      await renameDocument(id, title);
      setDocs(await listDocuments(folder).catch(() => docs));
    });
  };

  const handleAsk = () =>
    run("Thinking…", async () => {
      setAnswer(await query(question, folder));
    });

  if (!signedIn) {
    return (
      <>
        <h1>📖 ScholarRAG</h1>
        <button onClick={handleSignIn}>Sign in with Google</button>
        <p className="status">{status}</p>
      </>
    );
  }

  return (
    <>
      <div className="header">
        <h1>📖 ScholarRAG</h1>
        <button className="link" onClick={() => void signOut().then(refresh)}>
          Sign out
        </button>
      </div>

      <div className="label">Your folders</div>
      <div className="folders">
        {folders.map((f) => (
          <button
            key={f.name}
            className={`chip${f.name === folder ? " chip--active" : ""}`}
            onClick={() => onFolderChange(f.name)}
            title={f.count === null ? f.name : `${f.count} document${f.count === 1 ? "" : "s"}`}
          >
            <span className="chip__name">{f.name}</span>
            {f.count !== null && <span className="chip__count">{f.count}</span>}
          </button>
        ))}
        {creating ? (
          <input
            className="chip chip--new-input"
            autoFocus
            value={newFolder}
            placeholder="name…"
            onChange={(e) => setNewFolder(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") commitNewFolder();
              if (e.key === "Escape") {
                setCreating(false);
                setNewFolder("");
              }
            }}
            onBlur={commitNewFolder}
          />
        ) : (
          <button className="chip chip--new" onClick={() => setCreating(true)}>
            + New
          </button>
        )}
      </div>

      <input
        className="name-box"
        value={pageTitle}
        onChange={(e) => setPageTitle(e.target.value)}
        placeholder="Name for this page…"
      />
      <button className="add" onClick={handleAddPage}>
        + Add this page to “{folder}”
      </button>

      {docs.length > 0 && (
        <div className="docs">
          {docs.map((d) => (
            <div key={d.id} className="doc">
              {editingId === d.id ? (
                <input
                  className="doc__edit"
                  autoFocus
                  value={editName}
                  onChange={(e) => setEditName(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter") commitRename(d.id);
                    if (e.key === "Escape") setEditingId(null);
                  }}
                  onBlur={() => commitRename(d.id)}
                />
              ) : (
                <span
                  className="doc__name"
                  title={`${d.filename} — click to rename`}
                  onClick={() => startRename(d)}
                >
                  {d.filename}
                </span>
              )}
              <span className={`doc__status doc__status--${d.status}`}>{d.status}</span>
              <button className="doc__del" title="Remove from folder" onClick={() => handleDelete(d.id)}>
                ×
              </button>
            </div>
          ))}
        </div>
      )}

      <div className="row" style={{ marginTop: 12 }}>
        <input
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && handleAsk()}
          placeholder={`Ask about “${folder}”…`}
        />
        <button onClick={handleAsk} disabled={!question.trim()}>
          Ask
        </button>
      </div>

      <p className="status">{status}</p>

      {answer && (
        <div>
          <div className="answer">{answer.answer}</div>
          {answer.sources.length > 0 && (
            <div className="sources">
              Sources: {[...new Set(answer.sources.map((s) => s.filename))].join(", ")}
            </div>
          )}
        </div>
      )}
    </>
  );
}
