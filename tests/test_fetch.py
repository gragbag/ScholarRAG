"""URL fetch + content-type sniff (Phase 8, Step 2, exercise 2). Hermetic (httpx
MockTransport) — no network. ``httpx`` is core, so no extra needed."""

from __future__ import annotations

import httpx


# ── Exercise 2 — content-type sniffing (ingestion/fetch.py) ──────────────────
def test_sniff_extension() -> None:
    from scholarrag.ingestion.fetch import sniff_extension

    # header says PDF even though the URL path doesn't
    assert sniff_extension("https://arxiv.org/pdf/2301.00001", "application/pdf") == ".pdf"
    # URL path says PDF (case-insensitive) even with no header
    assert sniff_extension("https://x.example/paper.PDF", None) == ".pdf"
    # ordinary web page
    assert sniff_extension("https://blog.example/post", "text/html; charset=utf-8") == ".html"
    # default when nothing signals PDF
    assert sniff_extension("https://example.com/", None) == ".html"


def test_fetch_url_returns_bytes_and_filename() -> None:
    from scholarrag.ingestion.fetch import fetch_url

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, content=b"%PDF-1.4 fake", headers={"content-type": "application/pdf"}
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    data, filename = fetch_url("https://arxiv.org/pdf/2301.00001", client=client)
    assert data == b"%PDF-1.4 fake"
    assert filename.endswith(".pdf")  # sniffed from the content-type header
