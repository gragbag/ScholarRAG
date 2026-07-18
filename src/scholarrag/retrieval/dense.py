"""Dense (semantic) retrieval — nearest vectors in the embedding space.

Embeds the query with the *same* embedder used at ingestion, queries the vector
store, and maps the matches back to :class:`RetrievedChunk`s using the metadata
we stored on each vector (text, document_id, chunk_index, filename).
"""

from __future__ import annotations

import uuid

from sqlalchemy.orm import Session

from scholarrag.embeddings.base import Embedder
from scholarrag.retrieval.base import RetrievedChunk
from scholarrag.vectorstore.base import VectorStore


class DenseRetriever:
    """Semantic retrieval via the embedder + vector store."""

    def __init__(self, *, embedder: Embedder, vector_store: VectorStore) -> None:
        self._embedder = embedder
        self._vector_store = vector_store

    def retrieve(self, session: Session, query: str, *, top_k: int = 10) -> list[RetrievedChunk]:
        """Embed the query and return the ``top_k`` nearest chunks.

        ── YOUR TURN (Phase 2, Step 1 — exercise A) ────────────────────────────
        (``session`` is unused here — dense retrieval reads from the vector store.)

        1. Embed the query with the query-side method:
               vector = self._embedder.embed_query(query)
           (Not embed_documents — queries get the instruction prefix; this must be
           the same embedder that produced the stored vectors.)
        2. Query the store:
               matches = self._vector_store.query(vector, top_k=top_k)
        3. Map each match to a RetrievedChunk. Each ``match`` has ``.id`` (the
           vector_id), ``.score`` (cosine), and ``.metadata`` (a dict) carrying
           "text", "document_id", "chunk_index", "filename". Remember metadata
           values are typed loosely, so coerce:
               document_id=uuid.UUID(str(match.metadata["document_id"]))
               chunk_index=int(match.metadata["chunk_index"])
               text=str(match.metadata["text"]) ...
           Return the list (already ordered best-first by the store).

        Target test: ``test_dense_retriever_ranks_by_meaning`` in
        tests/test_retrieval.py. See EXERCISES.md → Phase 2 Step 1.
        ────────────────────────────────────────────────────────────────────────
        """
        vector = self._embedder.embed_query(query)

        matches = self._vector_store.query(vector, top_k=top_k)

        chunks = []
        for match in matches:
            chunk = RetrievedChunk(
                id=match.id,
                document_id=uuid.UUID(str(match.metadata["document_id"])),
                chunk_index=int(match.metadata["chunk_index"]), # type: ignore[arg-type]
                text=str(match.metadata["text"]),
                filename=str(match.metadata["filename"]),
                score=match.score,
            )
            chunks.append(chunk)
        
        return chunks
