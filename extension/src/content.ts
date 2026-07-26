// Content script — runs INSIDE every web page. When the popup sends an "EXTRACT"
// message, it runs Mozilla Readability (the engine behind Firefox Reader View) on
// the page and returns just the clean article text, dropping nav/ads/boilerplate.

import { Readability } from "@mozilla/readability";

chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
  if (msg?.type !== "EXTRACT") return;
  try {
    // Readability mutates the DOM, so parse a CLONE — never the live page.
    const doc = document.cloneNode(true) as Document;
    const article = new Readability(doc).parse();
    sendResponse({
      ok: true,
      title: article?.title ?? document.title,
      text: article?.textContent?.trim() ?? "",
    });
  } catch (e) {
    sendResponse({ ok: false, error: String(e) });
  }
  return true; // keep the channel open for the async sendResponse
});
