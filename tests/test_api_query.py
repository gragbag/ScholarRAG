"""API test for the /query route — hermetic via dependency overrides.

Overrides ``get_query_engine`` with a fake and ``get_db`` with a no-op, so the
route is exercised without an LLM key, retriever, or Postgres.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from typing import cast

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from scholarrag.api.deps import get_db, get_query_engine
from scholarrag.generation.base import Answer
from scholarrag.retrieval import RetrievedChunk


class _FakeEngine:
    def __init__(self, answer: Answer) -> None:
        self._answer = answer

    def query(self, session: Session, query: str) -> Answer:
        return self._answer

    def stream(self, session: Session, query: str) -> tuple[list[RetrievedChunk], Iterator[str]]:
        # candidate chunks + the answer streamed as one token, for the SSE route
        return self._answer.sources, iter([self._answer.text])


def _no_db() -> Iterator[None]:
    yield None


def test_query_endpoint_returns_answer_and_sources(client: TestClient) -> None:
    doc_id = uuid.uuid4()
    chunk = RetrievedChunk(
        id=f"{doc_id}:0",
        document_id=doc_id,
        chunk_index=0,
        text="RAG grounds answers in sources.",
        filename="rag.md",
        score=0.9,
    )
    fake = _FakeEngine(Answer(text="RAG grounds answers [1].", sources=[chunk]))

    app = cast(FastAPI, client.app)  # TestClient.app is typed as a bare ASGI callable
    app.dependency_overrides[get_query_engine] = lambda: fake
    app.dependency_overrides[get_db] = _no_db
    try:
        resp = client.post("/query", json={"query": "how does RAG work"})
    finally:
        app.dependency_overrides.pop(get_query_engine, None)
        app.dependency_overrides.pop(get_db, None)

    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "RAG grounds answers [1]."
    assert body["sources"][0]["filename"] == "rag.md"
    assert body["sources"][0]["document_id"] == str(doc_id)


def test_query_stream_endpoint_emits_tokens_then_sources(client: TestClient) -> None:
    doc_id = uuid.uuid4()
    chunk = RetrievedChunk(
        id=f"{doc_id}:0",
        document_id=doc_id,
        chunk_index=0,
        text="RAG grounds answers in sources.",
        filename="rag.md",
        score=0.9,
    )
    fake = _FakeEngine(Answer(text="RAG grounds answers [1].", sources=[chunk]))

    app = cast(FastAPI, client.app)
    app.dependency_overrides[get_query_engine] = lambda: fake
    app.dependency_overrides[get_db] = _no_db
    try:
        resp = client.post("/query/stream", json={"query": "how does RAG work"})
    finally:
        app.dependency_overrides.pop(get_query_engine, None)
        app.dependency_overrides.pop(get_db, None)

    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    body = resp.text
    assert '"type": "token"' in body
    assert "RAG grounds answers [1]." in body  # the streamed answer text
    assert '"type": "sources"' in body
    assert "rag.md" in body  # the cited source
    assert '"type": "done"' in body
