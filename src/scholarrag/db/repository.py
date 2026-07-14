"""Repository — typed, testable functions for reading/writing the tables.

Every function takes an explicit ``Session`` and only ``flush``es (see
``engine.py`` for why). Keeping SQL access here — rather than sprinkled through
routes and workers — means one place to reason about queries, and one place to
test them.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from scholarrag.db.models import Chunk, Document, IngestionStatus


@dataclass(frozen=True, slots=True)
class NewChunk:
    """Input for :func:`add_chunks` — the chunker (Step 3) produces these."""

    chunk_index: int
    text: str
    vector_id: str
    char_count: int


def create_document(
    session: Session,
    *,
    filename: str,
    content_hash: str,
    content_type: str,
    corpus_profile: str,
) -> Document:
    """Insert a new document row in the ``queued`` state and return it."""
    document = Document(
        filename=filename,
        content_hash=content_hash,
        content_type=content_type,
        corpus_profile=corpus_profile,
        status=IngestionStatus.queued,
    )
    session.add(document)
    session.flush()  # assigns document.id without committing
    return document


def get_document(session: Session, document_id: uuid.UUID) -> Document | None:
    """Fetch a document by primary key, or None."""
    return session.get(Document, document_id)


def get_document_by_hash(session: Session, content_hash: str) -> Document | None:
    """Idempotency lookup: has a document with these exact bytes been seen?"""
    stmt = select(Document).where(Document.content_hash == content_hash)
    return session.scalars(stmt).one_or_none()


def set_document_status(
    session: Session,
    document_id: uuid.UUID,
    status: IngestionStatus,
    *,
    error: str | None = None,
) -> None:
    """Update a document's ingestion status (and optional error message)."""
    document = session.get(Document, document_id)
    if document is None:
        raise ValueError(f"document {document_id} not found")
    document.status = status
    document.error = error
    session.flush()


def add_chunks(
    session: Session,
    document_id: uuid.UUID,
    new_chunks: list[NewChunk],
) -> int:
    """Bulk-insert chunks for a document and bump its ``num_chunks``.

    The ``fts`` tsvector is populated by Postgres automatically (generated
    column), so we never set it here.
    """
    rows = [
        Chunk(
            document_id=document_id,
            chunk_index=c.chunk_index,
            text=c.text,
            vector_id=c.vector_id,
            char_count=c.char_count,
        )
        for c in new_chunks
    ]
    session.add_all(rows)
    session.flush()

    document = session.get(Document, document_id)
    if document is not None:
        document.num_chunks = (document.num_chunks or 0) + len(rows)
        session.flush()
    return len(rows)


def count_chunks(session: Session, document_id: uuid.UUID) -> int:
    """Number of chunks stored for a document."""
    stmt = select(Chunk).where(Chunk.document_id == document_id)
    return len(session.scalars(stmt).all())


def list_documents(
    session: Session,
    *,
    limit: int = 50,
    offset: int = 0,
) -> list[Document]:
    "Return documents newest-first, paginated by ``limit`` / ``offset``."
    stmt = select(Document).order_by(Document.created_at.desc()).limit(limit).offset(offset)
    documents = list(session.scalars(stmt).all())
    return documents
