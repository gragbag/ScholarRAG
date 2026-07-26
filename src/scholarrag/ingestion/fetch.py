"""Fetch a remote page for ingestion, and decide how to treat it (PDF vs HTML).

Used by ``POST /documents/url``: given a URL, download the bytes and pick the
right extension so the ingestion pipeline runs the correct extractor. ``httpx``
is a core dependency, so this module needs no extra.
"""

from __future__ import annotations

from urllib.parse import urlparse

import httpx

MAX_FETCH_BYTES = 25 * 1024 * 1024  # 25 MB — same ceiling as file upload
_FETCH_TIMEOUT = httpx.Timeout(20.0, connect=5.0)


class FetchError(Exception):
    """The URL could not be fetched (network error, non-2xx, or too large)."""


def sniff_extension(url: str, content_type_header: str | None) -> str:
    "Decide the ingest extension — `.pdf` or `.html` — for a fetched URL."
    path = urlparse(url).path.lower()
    if path.endswith(".pdf"):
        return ".pdf"

    if content_type_header and "application/pdf" in content_type_header.lower():
        return ".pdf"

    return ".html"


def _base_filename(url: str) -> str:
    """A readable base name from a URL (last path segment, or the host)."""
    parsed = urlparse(url)
    last = parsed.path.rstrip("/").rsplit("/", 1)[-1]
    stem = last.rsplit(".", 1)[0]  # drop any existing extension; we add our own
    return stem or parsed.netloc or "page"


def fetch_url(url: str, *, client: httpx.Client | None = None) -> tuple[bytes, str]:
    """GET ``url`` and return ``(bytes, filename)`` with the sniffed extension.

    The ``client`` seam lets tests inject an ``httpx.MockTransport`` (no network).
    """
    owns_client = client is None
    client = client or httpx.Client(timeout=_FETCH_TIMEOUT, follow_redirects=True)
    try:
        resp = client.get(url)
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        raise FetchError(str(exc)) from exc
    finally:
        if owns_client:
            client.close()

    if len(resp.content) > MAX_FETCH_BYTES:
        raise FetchError(f"page exceeds {MAX_FETCH_BYTES} bytes")

    ext = sniff_extension(url, resp.headers.get("content-type"))
    return resp.content, _base_filename(url) + ext
