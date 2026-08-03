"""Document routes — upload (async ingest), status polling, and listing.

The upload endpoint is the *producer* side of the queue: it stores the file,
creates a ``queued`` document, enqueues the background task, and returns
immediately with ``202 Accepted`` + the id. Clients then poll ``GET
/documents/{id}`` to watch the status move ``queued → running → completed``.
"""

from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from scholarrag.api.deps import Enqueuer, get_db, get_enqueuer, get_pipeline
from scholarrag.auth.deps import get_current_user, get_current_user_optional
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
    # Optional display name. Without it, the name is the URL's last path segment
    # (e.g. "2603.10277v2" for arXiv) — so the extension lets the user set it.
    title: str | None = Field(default=None, max_length=256)


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
    if body.title:
        # Keep the sniffed extension (.pdf/.html) so content-type detection — which
        # decides how to parse — still works; the user's title is just the base name.
        filename = body.title + Path(filename).suffix
    return _register_and_enqueue(
        session, pipeline, enqueue, data=data, filename=filename, folder=body.folder, user=user
    )


@router.get("", response_model=list[DocumentResponse])
async def list_documents(
    session: Session = Depends(get_db),
    folder: str | None = None,
    limit: int = 50,
    offset: int = 0,
    user: User | None = Depends(get_current_user_optional),
) -> list[DocumentResponse]:
    """List documents newest-first, optionally scoped to a ``folder`` and to the
    caller (so a signed-in user sees only their own pages in that folder)."""
    documents = repo.list_documents(
        session,
        collection=folder,
        user_id=user.id if user is not None else None,
        limit=limit,
        offset=offset,
    )
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


class RenameRequest(BaseModel):
    """New display name for a document."""

    title: str = Field(min_length=1, max_length=256)


@router.patch("/{document_id}", response_model=DocumentResponse)
async def rename_document(
    document_id: uuid.UUID,
    body: RenameRequest,
    session: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> DocumentResponse:
    """Rename one of the caller's documents. 404 if it isn't theirs."""
    document = repo.rename_document(session, document_id, user_id=user.id, title=body.title)
    if document is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="document not found")
    return _to_response(document)


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    document_id: uuid.UUID,
    session: Session = Depends(get_db),
    pipeline: IngestionPipeline = Depends(get_pipeline),
    user: User = Depends(get_current_user),
) -> None:
    """Delete one of the caller's documents — its chunks (DB) and vectors (store).
    404 if it isn't theirs."""
    deleted = pipeline.delete_document(session, document_id, user_id=user.id)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="document not found")
