"""Database schema — the tables that track ingestion.

Two tables:

* ``documents`` — one row per uploaded file, carrying its ingestion status and
  a ``content_hash`` used for idempotency (re-uploading identical bytes is a
  no-op instead of a duplicate).
* ``chunks``    — the pieces a document is split into. Each chunk stores its
  text, the id it was given in the vector store, and a generated ``fts``
  column — a Postgres ``tsvector`` that is the lexical (BM25) search index.

The ``fts`` column is a *generated* column: Postgres computes
``to_tsvector('english', text)`` automatically on every insert/update, so the
full-text index can never drift out of sync with the text. A GIN index over it
makes lexical search fast.
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime

from sqlalchemy import (
    Computed,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy import (
    Enum as SAEnum,
)
from sqlalchemy.dialects.postgresql import TSVECTOR
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Declarative base — all models inherit from this; carries the metadata."""


class IngestionStatus(enum.StrEnum):
    """Lifecycle of a document as it moves through the ingestion pipeline."""

    queued = "queued"  # accepted, waiting for a worker
    running = "running"  # a worker is parsing/chunking/embedding it
    completed = "completed"  # vectors + chunks written successfully
    failed = "failed"  # a transient failure; may be retried
    dead_letter = "dead_letter"  # gave up after retries (poison document)


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    filename: Mapped[str] = mapped_column(String(1024))
    # sha256 of the raw file bytes — the idempotency key (unique).
    content_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    content_type: Mapped[str] = mapped_column(String(32))  # "pdf" | "md" | "txt"
    corpus_profile: Mapped[str] = mapped_column(String(64))
    status: Mapped[IngestionStatus] = mapped_column(
        SAEnum(
            IngestionStatus,
            name="ingestion_status",
            values_callable=lambda e: [m.value for m in e],
        ),
        default=IngestionStatus.queued,
        index=True,
    )
    error: Mapped[str | None] = mapped_column(Text, default=None)
    num_chunks: Mapped[int] = mapped_column(Integer, default=0)
    # Raw uploaded bytes, so a background worker (a separate process) can fetch
    # them by id. `deferred` keeps them out of ordinary SELECTs; Postgres stores
    # large values out-of-line via TOAST. (Object storage is the scale-up path.)
    raw_content: Mapped[bytes | None] = mapped_column(LargeBinary, deferred=True, default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    chunks: Mapped[list[Chunk]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Document(id={self.id!s}, filename={self.filename!r}, status={self.status.value})"


class Chunk(Base):
    __tablename__ = "chunks"
    __table_args__ = (
        # A document's chunk indexes are unique and ordered.
        UniqueConstraint("document_id", "chunk_index", name="uq_chunk_doc_index"),
        # GIN index makes `fts @@ to_tsquery(...)` lookups fast.
        Index("ix_chunks_fts", "fts", postgresql_using="gin"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        index=True,
    )
    chunk_index: Mapped[int] = mapped_column(Integer)  # order within the document
    text: Mapped[str] = mapped_column(Text)
    # The id this chunk was given in the vector store (maps PG <-> vectors).
    vector_id: Mapped[str] = mapped_column(String(128), index=True)
    char_count: Mapped[int] = mapped_column(Integer)
    # Generated column: Postgres keeps this tsvector in sync with `text`.
    fts: Mapped[str] = mapped_column(
        TSVECTOR, Computed("to_tsvector('english', text)", persisted=True)
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    document: Mapped[Document] = relationship(back_populates="chunks")

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Chunk(id={self.id!s}, document_id={self.document_id!s}, index={self.chunk_index})"
