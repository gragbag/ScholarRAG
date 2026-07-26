// Tiny wrappers over chrome.storage.local for the session token + current folder.
// chrome.storage is the extension's persistent key/value store (survives popup
// closes and browser restarts), unlike a webpage's localStorage.

const TOKEN_KEY = "scholarrag_jwt";
const FOLDER_KEY = "scholarrag_folder";

export async function getToken(): Promise<string | null> {
  const v = await chrome.storage.local.get(TOKEN_KEY);
  return (v[TOKEN_KEY] as string) ?? null;
}

export async function setToken(token: string): Promise<void> {
  await chrome.storage.local.set({ [TOKEN_KEY]: token });
}

export async function clearToken(): Promise<void> {
  await chrome.storage.local.remove(TOKEN_KEY);
}

export async function getFolder(): Promise<string> {
  const v = await chrome.storage.local.get(FOLDER_KEY);
  return (v[FOLDER_KEY] as string) ?? "default";
}

export async function setFolder(folder: string): Promise<void> {
  await chrome.storage.local.set({ [FOLDER_KEY]: folder });
}
