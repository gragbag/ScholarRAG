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
from scholarrag.auth.deps import get_current_user, get_current_user_optional
from scholarrag.db.models import User
from scholarrag.generation import cited_sources
from scholarrag.generation.base import Answer
from scholarrag.guardrails import sanitize_query
from scholarrag.pipeline import AnyQueryEngine
from scholarrag.retrieval.base import RetrievedChunk


def _owner_id(user: User | None) -> uuid.UUID | None:
    """The scope key: an authenticated user sees their own docs; None = anonymous."""
    return user.id if user is not None else None


router = APIRouter(tags=["query"])


class QueryRequest(BaseModel):
    # Length bounds are the first input guardrail: junk and oversized payloads
    # get a free 422 from the schema before any pipeline work happens.
    query: str = Field(min_length=3, max_length=2000)
    # Scope retrieval to one folder; omit (null) to search every folder.
    folder: str | None = Field(default=None, max_length=64)


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
    engine: AnyQueryEngine = Depends(get_query_engine),
    user: User | None = Depends(get_current_user_optional),
) -> QueryResponse:
    """Answer a question over the corpus, grounded in retrieved sources."""
    _check_rate_limit(http_request)
    answer = engine.query(
        session,
        sanitize_query(request.query),
        collection=request.folder,
        user_id=_owner_id(user),
    )
    return _to_response(answer)


@router.get("/folders", response_model=list[str], tags=["folders"])
async def list_folders(
    session: Session = Depends(get_db),
    user: User | None = Depends(get_current_user_optional),
) -> list[str]:
    """The caller's folders: the distinct collections on their documents, unioned
    with any empty folders they've created (the folders table)."""
    from scholarrag.db import repository as repo

    uid = _owner_id(user)
    names = set(repo.list_collections(session, user_id=uid))
    if uid is not None:
        names |= set(repo.list_user_folders(session, uid))
    return sorted(names)


class FolderSummary(BaseModel):
    """A folder and how many of the caller's documents live in it."""

    name: str
    count: int


@router.get("/folders/summary", response_model=list[FolderSummary], tags=["folders"])
async def list_folder_summaries(
    session: Session = Depends(get_db),
    user: User | None = Depends(get_current_user_optional),
) -> list[FolderSummary]:
    """The caller's folders, each with its document count (for the folder chips).
    Empty folders they've created show up with a count of 0."""
    from scholarrag.db import repository as repo

    uid = _owner_id(user)
    counts = dict(repo.folder_summaries(session, user_id=uid))  # folders that have docs
    if uid is not None:
        for name in repo.list_user_folders(session, uid):
            counts.setdefault(name, 0)  # created-but-empty folders → 0
    return [FolderSummary(name=name, count=count) for name, count in sorted(counts.items())]


class CreateFolderRequest(BaseModel):
    name: str = Field(min_length=1, max_length=64)


@router.post(
    "/folders",
    response_model=FolderSummary,
    status_code=status.HTTP_201_CREATED,
    tags=["folders"],
)
async def create_folder(
    body: CreateFolderRequest,
    session: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> FolderSummary:
    """Create (or return the existing) folder for the signed-in user. Requires auth
    — folders are owned, so there's no anonymous/public create."""
    from scholarrag.db import repository as repo

    folder = repo.create_folder(session, user_id=user.id, name=body.name)
    return FolderSummary(name=folder.name, count=0)


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
    engine: AnyQueryEngine = Depends(get_query_engine),
    user: User | None = Depends(get_current_user_optional),
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
        chunks, tokens = engine.stream(
            session,
            sanitize_query(request.query),
            collection=request.folder,
            user_id=_owner_id(user),
        )
        collected: list[str] = []
        for token in tokens:
            collected.append(token)
            yield _sse("token", {"text": token})
        sources = cited_sources("".join(collected), chunks)
        yield _sse("sources", {"sources": [_source_dict(c) for c in sources]})
        yield _sse("done", {})

    return StreamingResponse(event_gen(), media_type="text/event-stream")
