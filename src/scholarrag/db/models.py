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
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Computed,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy import (
    Enum as SAEnum,
)
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR
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


class User(Base):
    """An authenticated account. Populated on first Google sign-in."""

    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    # Google's stable subject id — the join key we trust (email can change).
    google_sub: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    email: Mapped[str] = mapped_column(String(320), index=True)
    name: Mapped[str | None] = mapped_column(String(255), default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"User(id={self.id!s}, email={self.email!r})"


class Folder(Base):
    """A user-created folder. Lets a folder persist while still empty — the folder
    list a user sees is the union of these rows and the distinct ``collection``
    values on their documents. Documents still reference a folder by its name
    string (``collection``), so there's no FK from documents to here."""

    __tablename__ = "folders"
    # A user can't have two folders with the same name (but two users can).
    __table_args__ = (UniqueConstraint("user_id", "name", name="uq_folders_user_name"),)

    id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        index=True,
    )
    name: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Folder(name={self.name!r}, user_id={self.user_id!s})"


class Document(Base):
    __tablename__ = "documents"
    # Idempotency is scoped to (owner, folder): the SAME bytes may exist once per
    # (user_id, collection) — so a user's copy of a paper is distinct from the
    # public seed corpus's copy, and the same paper can live in two folders. (For
    # public rows user_id is NULL, which Postgres treats as distinct — re-seed
    # idempotency is enforced by the application lookup, not this constraint.)
    __table_args__ = (
        UniqueConstraint(
            "user_id", "collection", "content_hash", name="uq_documents_owner_collection_hash"
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    filename: Mapped[str] = mapped_column(String(1024))
    # sha256 of the raw file bytes — the idempotency key, scoped per owner+folder
    # (see __table_args__). Indexed for the lookup, but no longer globally unique.
    content_hash: Mapped[str] = mapped_column(String(64), index=True)
    content_type: Mapped[str] = mapped_column(String(32))  # "pdf" | "md" | "txt" | "html"
    corpus_profile: Mapped[str] = mapped_column(String(64))
    # The user-facing "folder" a document belongs to. Retrieval can be scoped to
    # one collection. `corpus_profile` selects chunking behaviour; `collection`
    # groups documents — different concerns, so it's a separate column.
    collection: Mapped[str] = mapped_column(String(64), default="default", index=True)
    # The owner. NULL = public/seed corpus (visible to anonymous requests);
    # a user id = private to that user. Authenticated retrieval filters on it.
    user_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
        default=None,
    )
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
    # Key of the raw uploaded bytes in the blob store (object storage). The bytes
    # live there — not in Postgres — so the DB stays lean; a worker fetches by key.
    blob_key: Mapped[str | None] = mapped_column(String(256), default=None)
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


class Embedding(Base):
    """A stored chunk embedding — the consolidated (pgvector) vector store.

    Only the ``PgVectorStore`` backend reads/writes this table; Local/Pinecone
    ignore it. Keyed by the same ``vector_id`` that ``chunks.vector_id`` holds, so
    ``pipeline.delete_document`` purges vectors by id 1:1. The KNN index (HNSW)
    and the GIN index over ``metadata`` live in the migration, not here.
    """

    __tablename__ = "embeddings"

    # DIM is fixed at the BGE-small dimension (config.embedding_dim = 384). Change
    # both together if you swap the embedding model.
    vector_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    namespace: Mapped[str] = mapped_column(String(64), default="", server_default="")
    embedding: Mapped[Any] = mapped_column(Vector(384))
    # Filtered on with JSONB containment (@>) — {"collection": ..., "owner": ...}.
    meta: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSONB, default=dict, server_default="{}"
    )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Embedding(vector_id={self.vector_id!r}, ns={self.namespace!r})"
