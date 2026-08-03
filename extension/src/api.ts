// The backend API client. Every authenticated call attaches the session JWT as
// a Bearer header — the same token the backend's get_current_user reads to scope
// results to this user.

import { API_BASE } from "./config";
import { getToken } from "./storage";

async function authHeaders(): Promise<Record<string, string>> {
  const token = await getToken();
  const headers: Record<string, string> = { "content-type": "application/json" };
  if (token) headers["Authorization"] = `Bearer ${token}`;
  return headers;
}

export interface Source {
  document_id: string;
  filename: string;
  chunk_index: number;
  text: string;
}

export interface Answer {
  answer: string;
  sources: Source[];
}

/** POST /auth/google — exchange a Google ID token for our session JWT. */
export async function authGoogle(idToken: string): Promise<string> {
  const res = await fetch(`${API_BASE}/auth/google`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ id_token: idToken }),
  });
  if (!res.ok) throw new Error(`auth failed: ${res.status}`);
  const data = (await res.json()) as { access_token: string };
  return data.access_token;
}

/** GET /folders — the signed-in user's folders (names only). */
export async function listFolders(): Promise<string[]> {
  const res = await fetch(`${API_BASE}/folders`, { headers: await authHeaders() });
  if (!res.ok) throw new Error(`folders failed: ${res.status}`);
  return (await res.json()) as string[];
}

export interface FolderSummary {
  name: string;
  count: number;
}

/** GET /folders/summary — the user's folders, each with its document count. */
export async function listFolderSummaries(): Promise<FolderSummary[]> {
  const res = await fetch(`${API_BASE}/folders/summary`, { headers: await authHeaders() });
  if (!res.ok) throw new Error(`folder summary failed: ${res.status}`);
  return (await res.json()) as FolderSummary[];
}

export interface DocItem {
  id: string;
  filename: string;
  status: string;
  num_chunks: number;
}

/** GET /documents?folder= — the caller's pages in a folder, newest first. */
export async function listDocuments(folder: string): Promise<DocItem[]> {
  const url = `${API_BASE}/documents?folder=${encodeURIComponent(folder)}`;
  const res = await fetch(url, { headers: await authHeaders() });
  if (!res.ok) throw new Error(`documents failed: ${res.status}`);
  return (await res.json()) as DocItem[];
}

/** PATCH /documents/{id} — rename a page. */
export async function renameDocument(id: string, title: string): Promise<void> {
  const res = await fetch(`${API_BASE}/documents/${id}`, {
    method: "PATCH",
    headers: await authHeaders(),
    body: JSON.stringify({ title }),
  });
  if (!res.ok) throw new Error(`rename failed: ${res.status}`);
}

/** DELETE /documents/{id} — remove a page (and its vectors) from the folder. */
export async function deleteDocument(id: string): Promise<void> {
  const res = await fetch(`${API_BASE}/documents/${id}`, {
    method: "DELETE",
    headers: await authHeaders(),
  });
  if (!res.ok) throw new Error(`delete failed: ${res.status}`);
}

/** POST /folders — create (or return the existing) folder for the signed-in user. */
export async function createFolder(name: string): Promise<void> {
  const res = await fetch(`${API_BASE}/folders`, {
    method: "POST",
    headers: await authHeaders(),
    body: JSON.stringify({ name }),
  });
  if (!res.ok) throw new Error(`create folder failed: ${res.status}`);
}

/** POST /documents/text — ingest already-extracted page text into a folder. */
export async function addPageText(text: string, title: string, folder: string): Promise<void> {
  const res = await fetch(`${API_BASE}/documents/text`, {
    method: "POST",
    headers: await authHeaders(),
    body: JSON.stringify({ text, title, folder }),
  });
  if (!res.ok) throw new Error(`add page failed: ${res.status}`);
}

/** POST /documents/url — let the server fetch + ingest a URL (PDFs, or pages
 *  Readability can't handle). Used as the fallback for PDF tabs like arXiv.
 *  An optional title overrides the URL-derived name (e.g. arXiv ids). */
export async function addPageUrl(url: string, folder: string, title?: string): Promise<void> {
  const res = await fetch(`${API_BASE}/documents/url`, {
    method: "POST",
    headers: await authHeaders(),
    body: JSON.stringify({ url, folder, title }),
  });
  if (!res.ok) throw new Error(`add url failed: ${res.status}`);
}

/** POST /query — ask a question, scoped to a folder. */
export async function query(question: string, folder: string): Promise<Answer> {
  const res = await fetch(`${API_BASE}/query`, {
    method: "POST",
    headers: await authHeaders(),
    body: JSON.stringify({ query: question, folder }),
  });
  if (!res.ok) throw new Error(`query failed: ${res.status}`);
  return (await res.json()) as Answer;
}
