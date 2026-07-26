// The popup UI. It's wired to the API + auth + page-extraction helpers — but the
// LOOK is yours: restyle it, rearrange it, make it your own. It only works once
// you've done the three exercises (manifest permissions, signIn, extractActivePage).

import { useEffect, useState } from "react";
import { addPageText, addPageUrl, listFolders, query, type Answer } from "./api";
import { signIn, signOut } from "./auth";
import { extractActivePage } from "./page";
import { getFolder, getToken, setFolder } from "./storage";

export function Popup() {
  const [signedIn, setSignedIn] = useState(false);
  const [folders, setFolders] = useState<string[]>([]);
  const [folder, setFolderState] = useState("default");
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState<Answer | null>(null);
  const [status, setStatus] = useState("");

  async function refresh() {
    const token = await getToken();
    setSignedIn(!!token);
    setFolderState(await getFolder());
    if (token) {
      try {
        setFolders(await listFolders());
      } catch {
        /* not signed in yet, or backend down */
      }
    }
  }

  useEffect(() => {
    void refresh();
  }, []);

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

  const handleAddPage = () =>
    run("Adding this page…", async () => {
      const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

      // Try client-side Readability first. On a PDF page (arXiv, etc.) there is
      // no article DOM — extraction comes back empty, or the content script was
      // never injected into the PDF viewer — so we fall back to letting the
      // server fetch the URL (it sniffs application/pdf → runs the PDF pipeline).
      let title = tab?.title ?? "page";
      let text = "";
      try {
        const page = await extractActivePage();
        title = page.title || title;
        text = page.text;
      } catch {
        // no content script on this page (PDF viewer / chrome:// / store page)
      }

      if (text.trim().length > 50) {
        await addPageText(text, title, folder);
      } else if (tab?.url) {
        await addPageUrl(tab.url, folder); // PDF / unextractable → server fetches
      } else {
        throw new Error("Nothing to add — couldn't read this page.");
      }

      setStatus(`Added "${title}" to ${folder}`);
      setFolders(await listFolders());
    });

  const handleAsk = () =>
    run("Thinking…", async () => {
      setAnswer(await query(question, folder));
    });

  const onFolderChange = (value: string) => {
    setFolderState(value);
    void setFolder(value);
  };

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
      <h1>📖 ScholarRAG</h1>

      <div className="row">
        <input
          list="folders"
          value={folder}
          onChange={(e) => onFolderChange(e.target.value)}
          placeholder="folder"
        />
        <datalist id="folders">
          {folders.map((f) => (
            <option key={f} value={f} />
          ))}
        </datalist>
        <button onClick={handleAddPage}>Add page</button>
      </div>

      <div className="row">
        <input
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && handleAsk()}
          placeholder="Ask about this folder…"
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

      <button onClick={() => void signOut().then(refresh)} style={{ marginTop: 10, background: "#666" }}>
        Sign out
      </button>
    </>
  );
}
