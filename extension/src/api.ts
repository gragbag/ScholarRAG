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

/** GET /folders — the signed-in user's folders. */
export async function listFolders(): Promise<string[]> {
  const res = await fetch(`${API_BASE}/folders`, { headers: await authHeaders() });
  if (!res.ok) throw new Error(`folders failed: ${res.status}`);
  return (await res.json()) as string[];
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
 *  Readability can't handle). Used as the fallback for PDF tabs like arXiv. */
export async function addPageUrl(url: string, folder: string): Promise<void> {
  const res = await fetch(`${API_BASE}/documents/url`, {
    method: "POST",
    headers: await authHeaders(),
    body: JSON.stringify({ url, folder }),
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
