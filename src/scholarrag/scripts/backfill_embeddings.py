"""Backfill the pgvector ``embeddings`` table from existing chunks.

A one-time migration for when you switch ``VECTOR_STORE`` to pgvector *after*
documents were already ingested somewhere else (e.g. Pinecone): the ``chunks``
rows (text + the BM25 ``fts`` index) still exist, but the ``embeddings`` table is
empty — so DENSE retrieval finds nothing and only keyword queries get answered.

This re-embeds every chunk and upserts it into the configured vector store with
the SAME ``vector_id`` and the exact metadata ingestion writes (so retrieval can
rebuild each chunk, and folder/owner scoping still works). Idempotent — the store
upserts ON CONFLICT, so it's safe to re-run.

    make backfill-embeddings
"""

from __future__ import annotations

from sqlalchemy import select

from scholarrag.config import get_settings
from scholarrag.db.engine import session_scope
from scholarrag.db.models import Chunk, Document
from scholarrag.embeddings import build_embedder
from scholarrag.vectorstore import VectorRecord, build_vector_store
from scholarrag.vectorstore.base import Metadata


def main() -> None:
    settings = get_settings()
    embedder = build_embedder(settings)
    store = build_vector_store(settings)

    with session_scope() as session:
        rows = session.execute(
            select(
                Chunk.vector_id,
                Chunk.text,
                Chunk.chunk_index,
                Document.id,
                Document.filename,
                Document.collection,
                Document.user_id,
            ).join(Document, Chunk.document_id == Document.id)
        ).all()

    if not rows:
        print("no chunks to backfill")
        return

    print(f"embedding {len(rows)} chunks with {settings.embedding_model}…")
    vectors = embedder.embed_documents([row.text for row in rows])

    records = []
    for row, vector in zip(rows, vectors, strict=True):
        metadata: Metadata = {
            "text": row.text,
            "document_id": str(row.id),
            "chunk_index": row.chunk_index,
            "filename": row.filename,
            "collection": row.collection,
        }
        if row.user_id is not None:  # public/seed chunks stay unowned
            metadata["owner"] = str(row.user_id)
        records.append(VectorRecord(id=row.vector_id, values=vector, metadata=metadata))

    written = store.upsert(records)
    print(f"backfilled {written} embeddings into vector store: {settings.vector_store}")


if __name__ == "__main__":
    main()
