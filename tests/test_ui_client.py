"""UI client tests — hermetic (httpx.MockTransport, no network, no backend).

``httpx`` is a core dependency, so these run in CI without the ``ui`` extra.
One test passes now; the five skipped ones are the Phase 6 exercise targets.
"""

from __future__ import annotations

import httpx
import pytest

from scholarrag.ui.client import (
    Answer,
    RateLimited,
    ScholarRAGClient,
    Source,
    SSEEvent,
    StreamSink,
    api_url,
    iter_answer_tokens,
    parse_sse_stream,
)


# ── pass now ─────────────────────────────────────────────────────────────────
def test_api_url_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SCHOLARRAG_API_URL", raising=False)
    assert api_url() == "http://localhost:8001"
    monkeypatch.setenv("SCHOLARRAG_API_URL", "http://example.com:9000/")
    assert api_url() == "http://example.com:9000"  # trailing slash trimmed


def _client(handler: object) -> ScholarRAGClient:
    transport = httpx.MockTransport(handler)  # type: ignore[arg-type]
    return ScholarRAGClient(client=httpx.Client(base_url="http://test", transport=transport))


def test_parse_sse_stream() -> None:
    lines = [
        'data: {"type": "token", "text": "Hello"}',
        "",
        'data: {"type": "token", "text": " world"}',
        "",
        ": this is an SSE comment, ignore it",
        'data: {"type": "sources", "sources": []}',
        "",
        'data: {"type": "done"}',
        "",
    ]
    events = list(parse_sse_stream(lines))
    assert [e.type for e in events] == ["token", "token", "sources", "done"]
    assert events[0].data["text"] == "Hello"
    assert events[2].data["sources"] == []


def test_iter_answer_tokens() -> None:
    source = {"document_id": "d1", "filename": "rag.md", "chunk_index": 2, "text": "ctx"}
    events = [
        SSEEvent("token", {"type": "token", "text": "The "}),
        SSEEvent("token", {"type": "token", "text": "answer [1]"}),
        SSEEvent("sources", {"type": "sources", "sources": [source]}),
        SSEEvent("done", {"type": "done"}),
    ]
    sink = StreamSink()
    tokens = list(iter_answer_tokens(events, sink))

    assert tokens == ["The ", "answer [1]"]  # only token text, never sources
    assert sink.sources == [Source("d1", "rag.md", 2, "ctx")]
    assert sink.ungrounded is False


def test_iter_answer_tokens_flags_ungrounded() -> None:
    events = [
        SSEEvent("token", {"type": "token", "text": "unsupported claim"}),
        SSEEvent("sources", {"type": "sources", "sources": []}),  # empty = ungrounded
        SSEEvent("done", {"type": "done"}),
    ]
    sink = StreamSink()
    list(iter_answer_tokens(events, sink))
    assert sink.sources == []
    assert sink.ungrounded is True


def test_query_returns_answer() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/query"
        return httpx.Response(
            200,
            json={
                "answer": "grounded [1]",
                "sources": [
                    {"document_id": "d1", "filename": "rag.md", "chunk_index": 0, "text": "t"}
                ],
            },
        )

    answer = _client(handler).query("what is rag?")
    assert isinstance(answer, Answer)
    assert answer.text == "grounded [1]"
    assert answer.sources == [Source("d1", "rag.md", 0, "t")]


def test_query_maps_rate_limit() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, headers={"Retry-After": "60"}, json={"detail": "slow down"})

    with pytest.raises(RateLimited) as excinfo:
        _client(handler).query("too many")
    assert excinfo.value.retry_after == 60
