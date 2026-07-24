"""HTTP client for the ScholarRAG backend — what the Streamlit UI talks to.

The UI is a thin *client*: it owns no pipeline, no models, no database. It makes
HTTP calls to the FastAPI backend (default ``http://localhost:8000``) and renders
what comes back. That decoupling is deliberate — the same API could serve a React
app or a browser extension without changing a line here.

This module is pure logic (httpx + parsing) and fully unit-tested; the Streamlit
script in ``app.py`` is the un-testable glue layered on top of it. ``httpx`` is a
core dependency, so these tests run in CI without the ``ui`` extra.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any

import httpx

DEFAULT_API_URL = "http://localhost:8001"  # matches `make run` (uvicorn --port 8001)
_DEFAULT_TIMEOUT = httpx.Timeout(30.0, connect=5.0)
_STREAM_TIMEOUT = httpx.Timeout(90.0, connect=5.0)  # generation can be slow


def api_url() -> str:
    """Backend base URL from the environment (``SCHOLARRAG_API_URL``)."""
    return os.environ.get("SCHOLARRAG_API_URL", DEFAULT_API_URL).rstrip("/")


# ── wire types (mirror the backend's QueryResponse / SSE events) ─────────────
@dataclass(frozen=True, slots=True)
class Source:
    """One cited chunk, as the backend returns it."""

    document_id: str
    filename: str
    chunk_index: int
    text: str


@dataclass(frozen=True, slots=True)
class Answer:
    """A non-streaming answer: the text plus the sources it cited."""

    text: str
    sources: list[Source]


@dataclass(frozen=True, slots=True)
class SSEEvent:
    """One decoded Server-Sent Event: its ``type`` and the full JSON payload."""

    type: str
    data: dict[str, Any]


@dataclass
class StreamSink:
    """Mutable holder the streaming generator fills as a side effect.

    ``st.write_stream`` consumes a generator of *token strings* and returns the
    joined text — it can't also hand back the sources. So the generator yields
    text and drops the sources (and the ungrounded flag) in here for the caller
    to read once the stream is done.
    """

    sources: list[Source] = field(default_factory=list)
    ungrounded: bool = False


class RateLimited(Exception):
    """Backend returned 429 — the per-client rate limit is spent."""

    def __init__(self, retry_after: int | None = None) -> None:
        self.retry_after = retry_after
        super().__init__("Rate limit exceeded — wait a moment and try again.")


class BackendUnavailable(Exception):
    """The backend couldn't be reached (is ``make run`` up?)."""


def _to_source(d: dict[str, Any]) -> Source:
    return Source(
        document_id=str(d["document_id"]),
        filename=d["filename"],
        chunk_index=int(d["chunk_index"]),
        text=d["text"],
    )


def _retry_after(resp: httpx.Response) -> int | None:
    value = resp.headers.get("Retry-After")
    return int(value) if value and value.isdigit() else None


def parse_sse_stream(lines: Iterable[str]) -> Iterator[SSEEvent]:
    "Turn raw Server-Sent-Event lines into typed :class:`SSEEvent` objects."
    for line in lines:
        if not line or not line.startswith("data:"):
            continue

        payload = json.loads(line[5:])
        yield SSEEvent(type=payload["type"], data=payload)


def iter_answer_tokens(events: Iterable[SSEEvent], sink: StreamSink) -> Iterator[str]:
    "Yield answer text token-by-token; stash the sources into ``sink`` at the end."

    for event in events:
        if event.type == "token":
            yield event.data["text"]
        elif event.type == "sources":
            sink.sources = [_to_source(s) for s in event.data["sources"]]
            sink.ungrounded = not sink.sources
        elif event.type == "done":
            return


def stream_answer(client: httpx.Client, question: str, sink: StreamSink) -> Iterator[str]:
    """Open ``/query/stream`` and yield answer tokens (for ``st.write_stream``).

    Scaffolded glue: it wires ``client.stream`` → :func:`parse_sse_stream` →
    :func:`iter_answer_tokens`. Errors are mapped to the UI's typed exceptions.
    (This is a generator, so the request isn't sent until iteration begins — which
    is why the app's ``try/except`` around ``st.write_stream`` catches these.)
    """
    try:
        with client.stream(
            "POST", "/query/stream", json={"query": question}, timeout=_STREAM_TIMEOUT
        ) as resp:
            if resp.status_code == httpx.codes.TOO_MANY_REQUESTS:
                raise RateLimited(_retry_after(resp))
            resp.raise_for_status()
            yield from iter_answer_tokens(parse_sse_stream(resp.iter_lines()), sink)
    except httpx.ConnectError as exc:
        raise BackendUnavailable(str(exc)) from exc


class ScholarRAGClient:
    """Holds one ``httpx.Client`` and exposes the backend's two query endpoints."""

    def __init__(self, base_url: str | None = None, *, client: httpx.Client | None = None) -> None:
        # `client` injection point exists for tests (httpx.MockTransport).
        self._client = client or httpx.Client(
            base_url=base_url or api_url(), timeout=_DEFAULT_TIMEOUT
        )

    def close(self) -> None:
        self._client.close()

    def query(self, question: str) -> Answer:
        "POST ``/query`` and return the :class:`Answer`; map 429 → RateLimited."
        try:
            resp = self._client.post("/query", json={"query": question})
        except httpx.ConnectError as exc:
            raise BackendUnavailable(str(exc)) from exc

        if resp.status_code == httpx.codes.TOO_MANY_REQUESTS:
            raise RateLimited(_retry_after(resp))

        resp.raise_for_status()
        body = resp.json()

        return Answer(text=body["answer"], sources=[_to_source(s) for s in body["sources"]])

    def stream(self, question: str, sink: StreamSink) -> Iterator[str]:
        """Streaming query — delegates to :func:`stream_answer`."""
        return stream_answer(self._client, question, sink)
