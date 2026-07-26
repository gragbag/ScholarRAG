// Bridge from the popup to the content script: get the readable text of the tab
// the user is currently looking at.

export interface ExtractedPage {
  title: string;
  text: string;
}

export async function extractActivePage(): Promise<ExtractedPage> {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (!tab?.id) throw new Error("no active tab");

  const res = await chrome.tabs.sendMessage(tab.id, { type: "EXTRACT" });
  if (!res?.ok) throw new Error(res?.error ?? "extraction failed");
  return { title: res.title, text: res.text };
}
