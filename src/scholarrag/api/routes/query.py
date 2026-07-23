"""Query route — the RAG endpoint: ask a question, get a cited answer.

``POST /query`` runs the full pipeline (rewrite -> retrieve -> fuse -> generate)
and returns the grounded answer plus the sources it cited. The engine is injected
via ``Depends`` so tests swap in a fake — no LLM key, no Postgres, no models.
Streaming (``/query/stream``) lands in Step 4b.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterator

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from scholarrag.api.deps import get_db, get_query_engine
from scholarrag.generation import cited_sources
from scholarrag.generation.base import Answer
from scholarrag.guardrails import sanitize_query
from scholarrag.pipeline import QueryEngine
from scholarrag.retrieval.base import RetrievedChunk

router = APIRouter(tags=["query"])


class QueryRequest(BaseModel):
    # Length bounds are the first input guardrail: junk and oversized payloads
    # get a free 422 from the schema before any pipeline work happens.
    query: str = Field(min_length=3, max_length=2000)


def _check_rate_limit(request: Request) -> None:
    """Reject with 429 when the per-client budget is spent (no-op when disabled)."""
    limiter = request.app.state.rate_limiter
    if limiter is None:
        return
    client_id = request.client.host if request.client else "unknown"
    if not limiter.allow(client_id):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="rate limit exceeded — try again shortly",
            headers={"Retry-After": "60"},
        )


class SourceResponse(BaseModel):
    document_id: uuid.UUID
    filename: str
    chunk_index: int
    text: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceResponse]


def _to_response(answer: Answer) -> QueryResponse:
    return QueryResponse(
        answer=answer.text,
        sources=[
            SourceResponse(
                document_id=c.document_id,
                filename=c.filename,
                chunk_index=c.chunk_index,
                text=c.text,
            )
            for c in answer.sources
        ],
    )


@router.post("/query", response_model=QueryResponse)
async def query_documents(
    request: QueryRequest,
    http_request: Request,
    session: Session = Depends(get_db),
    engine: QueryEngine = Depends(get_query_engine),
) -> QueryResponse:
    """Answer a question over the corpus, grounded in retrieved sources."""
    _check_rate_limit(http_request)
    answer = engine.query(session, sanitize_query(request.query))
    return _to_response(answer)


def _source_dict(chunk: RetrievedChunk) -> dict[str, object]:
    return {
        "document_id": str(chunk.document_id),
        "filename": chunk.filename,
        "chunk_index": chunk.chunk_index,
        "text": chunk.text,
    }


def _sse(event_type: str, payload: dict[str, object]) -> str:
    """Frame one Server-Sent Event: a ``data:`` line of JSON, blank line terminates it."""
    return f"data: {json.dumps({'type': event_type, **payload})}\n\n"


@router.post("/query/stream")
async def query_stream(
    request: QueryRequest,
    http_request: Request,
    session: Session = Depends(get_db),
    engine: QueryEngine = Depends(get_query_engine),
) -> StreamingResponse:
    """Stream the grounded answer token-by-token as SSE, then emit the cited sources.

    Event sequence: many ``token`` events → one ``sources`` event (resolved from the
    citations in the full answer) → a final ``done`` event.

    Guardrail note: the grounding gate can't un-stream tokens, so this path is
    only gated post-hoc — an empty ``sources`` event is the client's signal that
    the streamed text was ungrounded. (Real systems sometimes buffer-then-stream
    for exactly this reason.)
    """
    _check_rate_limit(http_request)

    def event_gen() -> Iterator[str]:
        chunks, tokens = engine.stream(session, sanitize_query(request.query))
        collected: list[str] = []
        for token in tokens:
            collected.append(token)
            yield _sse("token", {"text": token})
        sources = cited_sources("".join(collected), chunks)
        yield _sse("sources", {"sources": [_source_dict(c) for c in sources]})
        yield _sse("done", {})

    return StreamingResponse(event_gen(), media_type="text/event-stream")
