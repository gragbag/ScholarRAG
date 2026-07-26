"""Document routes — upload (async ingest), status polling, and listing.

The upload endpoint is the *producer* side of the queue: it stores the file,
creates a ``queued`` document, enqueues the background task, and returns
immediately with ``202 Accepted`` + the id. Clients then poll ``GET
/documents/{id}`` to watch the status move ``queued → running → completed``.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from scholarrag.api.deps import Enqueuer, get_db, get_enqueuer, get_pipeline
from scholarrag.auth.deps import get_current_user_optional
from scholarrag.config import get_settings
from scholarrag.corpus import get_corpus_profile
from scholarrag.db import repository as repo
from scholarrag.db.models import Document, IngestionStatus, User
from scholarrag.ingestion import IngestionPipeline, UnsupportedFileTypeError
from scholarrag.ingestion.fetch import FetchError, fetch_url

router = APIRouter(prefix="/documents", tags=["documents"])

MAX_UPLOAD_BYTES = 25 * 1024 * 1024  # 25 MB


class UploadResponse(BaseModel):
    document_id: uuid.UUID
    status: IngestionStatus
    skipped: bool  # True if identical bytes were already ingested (not re-enqueued)


class DocumentResponse(BaseModel):
    id: uuid.UUID
    filename: str
    status: IngestionStatus
    num_chunks: int
    error: str | None


def _to_response(document: Document) -> DocumentResponse:
    return DocumentResponse(
        id=document.id,
        filename=document.filename,
        status=document.status,
        num_chunks=document.num_chunks,
        error=document.error,
    )


class TextIngestRequest(BaseModel):
    """Already-extracted readable text — the extension's Readability.js output."""

    text: str = Field(min_length=1, max_length=1_000_000)
    title: str = Field(default="page", max_length=256)
    folder: str = Field(default="default", max_length=64)


class UrlIngestRequest(BaseModel):
    """A URL for the server to fetch + ingest (PDFs, or an HTML fallback)."""

    url: str = Field(min_length=1, max_length=2048)
    folder: str = Field(default="default", max_length=64)


def _register_and_enqueue(
    session: Session,
    pipeline: IngestionPipeline,
    enqueue: Enqueuer,
    *,
    data: bytes,
    filename: str,
    folder: str,
    user: User | None,
) -> UploadResponse:
    """Register bytes into ``folder`` + enqueue ingestion — shared by every ingest route."""
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"content exceeds {MAX_UPLOAD_BYTES} bytes",
        )
    profile = get_corpus_profile(get_settings().corpus_profile)
    try:
        registration = pipeline.register(
            session,
            data=data,
            filename=filename,
            profile=profile,
            collection=folder,
            user_id=user.id if user is not None else None,
        )
    except UnsupportedFileTypeError as exc:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail=str(exc)
        ) from exc

    document = registration.document
    if registration.created:
        enqueue(document.id)  # hand off to a background worker
    return UploadResponse(
        document_id=document.id,
        status=document.status,
        skipped=not registration.created,
    )


@router.post("", status_code=status.HTTP_202_ACCEPTED, response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    folder: str = Form("default"),
    session: Session = Depends(get_db),
    pipeline: IngestionPipeline = Depends(get_pipeline),
    enqueue: Enqueuer = Depends(get_enqueuer),
    user: User | None = Depends(get_current_user_optional),
) -> UploadResponse:
    """Accept a file, register it into ``folder``, and enqueue background ingestion."""
    data = await file.read()
    return _register_and_enqueue(
        session,
        pipeline,
        enqueue,
        data=data,
        filename=file.filename or "upload",
        folder=folder,
        user=user,
    )


@router.post("/text", status_code=status.HTTP_202_ACCEPTED, response_model=UploadResponse)
async def ingest_text(
    body: TextIngestRequest,
    session: Session = Depends(get_db),
    pipeline: IngestionPipeline = Depends(get_pipeline),
    enqueue: Enqueuer = Depends(get_enqueuer),
    user: User | None = Depends(get_current_user_optional),
) -> UploadResponse:
    """Ingest already-clean text (the extension extracted it client-side)."""
    # `.txt` => the plain-text path (no re-extraction — the text is already clean).
    return _register_and_enqueue(
        session,
        pipeline,
        enqueue,
        data=body.text.encode("utf-8"),
        filename=f"{body.title}.txt",
        folder=body.folder,
        user=user,
    )


@router.post("/url", status_code=status.HTTP_202_ACCEPTED, response_model=UploadResponse)
async def ingest_url(
    body: UrlIngestRequest,
    session: Session = Depends(get_db),
    pipeline: IngestionPipeline = Depends(get_pipeline),
    enqueue: Enqueuer = Depends(get_enqueuer),
    user: User | None = Depends(get_current_user_optional),
) -> UploadResponse:
    """Fetch a URL server-side, then ingest it (PDF pipeline, or readable-HTML)."""
    try:
        data, filename = fetch_url(body.url)
    except FetchError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY, detail=f"could not fetch url: {exc}"
        ) from exc
    return _register_and_enqueue(
        session, pipeline, enqueue, data=data, filename=filename, folder=body.folder, user=user
    )


@router.get("", response_model=list[DocumentResponse])
async def list_documents(
    session: Session = Depends(get_db),
    limit: int = 50,
    offset: int = 0,
) -> list[DocumentResponse]:
    """List documents, newest first."""
    documents = repo.list_documents(session, limit=limit, offset=offset)
    return [_to_response(d) for d in documents]


@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: uuid.UUID,
    session: Session = Depends(get_db),
) -> DocumentResponse:
    "Return one document's ingestion status (poll this after uploading)."
    document = repo.get_document(session, document_id)

    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Document: {document_id}; not found"
        )

    return _to_response(document)
