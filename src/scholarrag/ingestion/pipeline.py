"""The ingestion pipeline — composing every piece into one idempotent flow.

``IngestionPipeline`` ties together the primitives you've built:

    hash → (idempotency check) → parse → chunk → embed → upsert vectors
         → write chunk rows → mark the document completed.

Two entry points, so the same code serves the sync and async paths:

* ``register`` — hash, idempotency check, create the ``queued`` document row and
  store its raw bytes. Fast; the API calls this then enqueues a task.
* ``process``  — load the stored document + bytes and run the heavy work. The
  Celery worker calls this by ``document_id`` (Step 5).
* ``ingest``   — ``register`` then ``process``, for the synchronous callers
  (tests, the seed script).

It's constructed with its *long-lived* dependencies (an ``Embedder`` and a
``VectorStore``) — dependency injection, so tests wire in ``FakeEmbedder`` +
``LocalVectorStore`` with no cloud or model download.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from sqlalchemy.orm import Session

from scholarrag.blobstore.base import BlobStore
from scholarrag.blobstore.memory import MemoryBlobStore
from scholarrag.corpus import CorpusProfile, get_corpus_profile
from scholarrag.db import repository as repo
from scholarrag.db.models import Document, IngestionStatus
from scholarrag.db.repository import NewChunk
from scholarrag.embeddings.base import Embedder, Vector
from scholarrag.ingestion.chunk import TextChunk, chunk_text
from scholarrag.ingestion.hashing import content_hash
from scholarrag.ingestion.parse import detect_content_type, extract_text
from scholarrag.vectorstore.base import Metadata, VectorRecord, VectorStore


class TransientIngestionError(Exception):
    """A *retryable* failure (network blip, timeout, a paused vector index).

    Anything that is NOT a ``TransientIngestionError`` is treated as permanent by
    the worker and sent straight to the dead-letter queue instead of retried.
    """


@dataclass(frozen=True, slots=True)
class IngestResult:
    """Outcome of an ingest/process call."""

    document_id: uuid.UUID
    status: IngestionStatus
    num_chunks: int
    skipped: bool  # True when an identical file had already been ingested


@dataclass(frozen=True, slots=True)
class RegisterResult:
    """Outcome of ``register``: the document and whether it was newly created."""

    document: Document
    created: bool  # False when an identical file was already registered


class IngestionPipeline:
    """Orchestrates ingestion. Holds the long-lived embedder + vector store."""

    def __init__(
        self,
        *,
        embedder: Embedder,
        vector_store: VectorStore,
        blob_store: BlobStore | None = None,
    ) -> None:
        self._embedder = embedder
        self._vector_store = vector_store
        # Default to in-memory (single-process / tests); prod passes a real store.
        self._blob_store = blob_store if blob_store is not None else MemoryBlobStore()

    def register(
        self,
        session: Session,
        *,
        data: bytes,
        filename: str,
        profile: CorpusProfile,
        collection: str = "default",
        user_id: uuid.UUID | None = None,
    ) -> RegisterResult:
        """Create a ``queued`` document + store its bytes (idempotent by hash).

        Returns the existing document with ``created=False`` if these exact bytes
        were already registered — so the caller knows not to enqueue again.
        """
        digest = content_hash(data)
        existing = self._find_existing(session, digest, collection=collection, user_id=user_id)
        if existing is not None:
            return RegisterResult(document=existing, created=False)

        content_type = detect_content_type(filename)
        document = repo.create_document(
            session,
            filename=filename,
            content_hash=digest,
            content_type=content_type,
            corpus_profile=profile.name,
            collection=collection,
            user_id=user_id,
        )
        # Raw bytes go to the blob store (object storage), not Postgres; the row
        # keeps only the key so a worker can fetch them later.
        document.blob_key = f"documents/{document.id}"
        self._blob_store.put(document.blob_key, data)
        session.flush()
        session.commit()
        return RegisterResult(document=document, created=True)

    def process(self, session: Session, document_id: uuid.UUID) -> IngestResult:
        """Run the heavy work for an already-registered document.

        Loads the document + its stored bytes, then parse → chunk → embed →
        upsert → write chunk rows → completed. On failure the document is marked
        ``failed`` and the error re-raised (the worker decides retry vs DLQ).
        """
        document = repo.get_document(session, document_id)
        if document is None:
            raise ValueError(f"document {document_id} not found")
        if document.blob_key is None:
            raise ValueError(f"document {document_id} has no stored content")
        profile = get_corpus_profile(document.corpus_profile)

        repo.set_document_status(session, document_id, IngestionStatus.running)
        session.commit()

        try:
            data = self._blob_store.get(document.blob_key)
            text = extract_text(data, document.content_type)
            chunks = chunk_text(text, profile)
            embeddings = self._embedder.embed_documents([c.text for c in chunks]) if chunks else []
            records, new_chunks = self._build_records(
                document_id,
                document.filename,
                document.collection,
                document.user_id,
                chunks,
                embeddings,
            )
            self._vector_store.upsert(records)
            repo.add_chunks(session, document_id, new_chunks)
            repo.set_document_status(session, document_id, IngestionStatus.completed)
            session.commit()
        except Exception as exc:
            session.rollback()
            repo.set_document_status(session, document_id, IngestionStatus.failed, error=str(exc))
            session.commit()
            raise

        return IngestResult(
            document_id=document_id,
            status=document.status,
            num_chunks=document.num_chunks,
            skipped=False,
        )

    def ingest(
        self,
        session: Session,
        *,
        data: bytes,
        filename: str,
        profile: CorpusProfile,
        collection: str = "default",
        user_id: uuid.UUID | None = None,
    ) -> IngestResult:
        """Synchronous end-to-end ingest (register + process)."""
        registration = self.register(
            session,
            data=data,
            filename=filename,
            profile=profile,
            collection=collection,
            user_id=user_id,
        )
        document = registration.document
        if not registration.created:
            return IngestResult(
                document_id=document.id,
                status=document.status,
                num_chunks=document.num_chunks,
                skipped=True,
            )
        return self.process(session, document.id)

    def delete_document(
        self, session: Session, document_id: uuid.UUID, *, user_id: uuid.UUID | None
    ) -> bool:
        """Delete a document, its chunks (DB cascade), AND its vectors (store). Scoped
        to the owner via the repository. Returns True if it was deleted.

        Order matters: gather the vector ids first, delete the row (owner-scoped),
        commit, then purge the vectors — so an unauthorized delete never touches the
        store, and the DB is durable before the external side effect.
        """
        vector_ids = repo.document_vector_ids(session, document_id)
        document = repo.get_document(session, document_id)
        blob_key = document.blob_key if document is not None else None
        deleted = repo.delete_document(session, document_id, user_id=user_id)
        if deleted:
            session.commit()
            if vector_ids:
                self._vector_store.delete(ids=vector_ids)
            if blob_key:
                self._blob_store.delete(blob_key)
        return deleted

    def _find_existing(
        self,
        session: Session,
        digest: str,
        *,
        collection: str = "default",
        user_id: uuid.UUID | None = None,
    ) -> Document | None:
        """Idempotency lookup — has THIS user already ingested these exact bytes
        into THIS folder? (Scoped so the same paper can live in two folders, and a
        user's copy is separate from the public seed corpus.)"""
        return repo.get_document_by_hash(session, digest, user_id=user_id, collection=collection)

    def _build_records(
        self,
        document_id: uuid.UUID,
        filename: str,
        collection: str,
        user_id: uuid.UUID | None,
        chunks: list[TextChunk],
        embeddings: list[Vector],
    ) -> tuple[list[VectorRecord], list[NewChunk]]:
        """Zip chunks with their embeddings into vector-store + DB rows."""
        records = []
        new_chunks = []
        for chunk, embedding in zip(chunks, embeddings, strict=True):
            vector_id = f"{document_id}:{chunk.index}"
            metadata: Metadata = {
                "text": chunk.text,
                "document_id": str(document_id),
                "chunk_index": chunk.index,
                "filename": filename,
                "collection": collection,  # dense retrieval scopes by folder
            }
            # Tag the owner only for user docs; public/seed docs stay unowned, so
            # anonymous retrieval (no owner filter) still finds them.
            if user_id is not None:
                metadata["owner"] = str(user_id)
            record = VectorRecord(id=vector_id, values=embedding, metadata=metadata)
            new_chunk = NewChunk(
                chunk_index=chunk.index,
                text=chunk.text,
                vector_id=vector_id,
                char_count=chunk.char_count,
            )

            records.append(record)
            new_chunks.append(new_chunk)
        return records, new_chunks
