"""The ingestion pipeline — composing every piece into one idempotent flow.

``IngestionPipeline`` ties together the primitives you've built:

    hash → (idempotency check) → parse → chunk → embed → upsert vectors
         → write chunk rows → mark the document completed.

It's constructed with its *long-lived* dependencies (an ``Embedder`` and a
``VectorStore``) and its ``ingest`` method takes the *per-request* session. That
separation is dependency injection: production wires in the real BGE embedder +
Pinecone; tests wire in ``FakeEmbedder`` + ``LocalVectorStore`` — same pipeline
code, no cloud, no model download.

Still synchronous — Step 5 wraps this in a Celery task for retries + a DLQ.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from sqlalchemy.orm import Session

from scholarrag.corpus import CorpusProfile
from scholarrag.db import repository as repo
from scholarrag.db.models import Document, IngestionStatus
from scholarrag.db.repository import NewChunk
from scholarrag.embeddings.base import Embedder, Vector
from scholarrag.ingestion.chunk import TextChunk, chunk_text
from scholarrag.ingestion.hashing import content_hash
from scholarrag.ingestion.parse import detect_content_type, extract_text
from scholarrag.vectorstore.base import VectorRecord, VectorStore


@dataclass(frozen=True, slots=True)
class IngestResult:
    """Outcome of an ingest call."""

    document_id: uuid.UUID
    status: IngestionStatus
    num_chunks: int
    skipped: bool  # True when an identical file had already been ingested


class IngestionPipeline:
    """Orchestrates ingestion. Holds the long-lived embedder + vector store."""

    def __init__(self, *, embedder: Embedder, vector_store: VectorStore) -> None:
        self._embedder = embedder
        self._vector_store = vector_store

    def ingest(
        self,
        session: Session,
        *,
        data: bytes,
        filename: str,
        profile: CorpusProfile,
    ) -> IngestResult:
        """Ingest one document end-to-end. Idempotent by content hash."""
        digest = content_hash(data)

        # Idempotency: identical bytes already ingested → skip.
        existing = self._find_existing(session, digest)
        if existing is not None:
            return IngestResult(
                document_id=existing.id,
                status=existing.status,
                num_chunks=existing.num_chunks,
                skipped=True,
            )

        content_type = detect_content_type(filename)
        document = repo.create_document(
            session,
            filename=filename,
            content_hash=digest,
            content_type=content_type,
            corpus_profile=profile.name,
        )
        # Commit `running` early so async callers (Step 5) can see progress.
        repo.set_document_status(session, document.id, IngestionStatus.running)
        session.commit()

        try:
            text = extract_text(data, content_type)
            chunks = chunk_text(text, profile)
            embeddings = self._embedder.embed_documents([c.text for c in chunks]) if chunks else []
            records, new_chunks = self._build_records(document.id, filename, chunks, embeddings)
            self._vector_store.upsert(records)
            repo.add_chunks(session, document.id, new_chunks)
            repo.set_document_status(session, document.id, IngestionStatus.completed)
            session.commit()
        except Exception as exc:
            # Roll back the partial work, then record the failure durably so the
            # document isn't left stuck in `running`. Re-raise for Step 5's retry.
            session.rollback()
            repo.set_document_status(session, document.id, IngestionStatus.failed, error=str(exc))
            session.commit()
            raise

        return IngestResult(
            document_id=document.id,
            status=document.status,
            num_chunks=document.num_chunks,
            skipped=False,
        )

    def _find_existing(self, session: Session, digest: str) -> Document | None:
        """Idempotency lookup — has a file with this exact hash been ingested?"""
        return repo.get_document_by_hash(session, digest)

    def _build_records(
        self,
        document_id: uuid.UUID,
        filename: str,
        chunks: list[TextChunk],
        embeddings: list[Vector],
    ) -> tuple[list[VectorRecord], list[NewChunk]]:
        "Zip chunks with their embeddings into vector-store + DB rows."

        records = []
        new_chunks = []
        for chunk, embedding in zip(chunks, embeddings, strict=True):
            vector_id = f"{document_id}:{chunk.index}"
            record = VectorRecord(
                id=vector_id,
                values=embedding,
                metadata={
                    "text": chunk.text,
                    "document_id": str(document_id),
                    "chunk_index": chunk.index,
                    "filename": filename,
                },
            )
            new_chunk = NewChunk(
                chunk_index=chunk.index,
                text=chunk.text,
                vector_id=vector_id,
                char_count=chunk.char_count,
            )

            records.append(record)
            new_chunks.append(new_chunk)
        return records, new_chunks
