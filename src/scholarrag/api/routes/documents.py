"""Document routes — upload (async ingest), status polling, and listing.

The upload endpoint is the *producer* side of the queue: it stores the file,
creates a ``queued`` document, enqueues the background task, and returns
immediately with ``202 Accepted`` + the id. Clients then poll ``GET
/documents/{id}`` to watch the status move ``queued → running → completed``.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from scholarrag.api.deps import Enqueuer, get_db, get_enqueuer, get_pipeline
from scholarrag.config import get_settings
from scholarrag.corpus import get_corpus_profile
from scholarrag.db import repository as repo
from scholarrag.db.models import Document, IngestionStatus
from scholarrag.ingestion import IngestionPipeline, UnsupportedFileTypeError

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


@router.post("", status_code=status.HTTP_202_ACCEPTED, response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    session: Session = Depends(get_db),
    pipeline: IngestionPipeline = Depends(get_pipeline),
    enqueue: Enqueuer = Depends(get_enqueuer),
) -> UploadResponse:
    """Accept a file, register it, and enqueue background ingestion."""
    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"file exceeds {MAX_UPLOAD_BYTES} bytes",
        )
    filename = file.filename or "upload"
    profile = get_corpus_profile(get_settings().corpus_profile)

    try:
        registration = pipeline.register(session, data=data, filename=filename, profile=profile)
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
