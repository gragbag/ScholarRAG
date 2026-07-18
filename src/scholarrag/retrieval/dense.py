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
        "Embed the query and return the ``top_k`` nearest chunks."
        vector = self._embedder.embed_query(query)

        matches = self._vector_store.query(vector, top_k=top_k)

        chunks = []
        for match in matches:
            chunk = RetrievedChunk(
                id=match.id,
                document_id=uuid.UUID(str(match.metadata["document_id"])),
                chunk_index=int(match.metadata["chunk_index"]),  # type: ignore[arg-type]
                text=str(match.metadata["text"]),
                filename=str(match.metadata["filename"]),
                score=match.score,
            )
            chunks.append(chunk)

        return chunks
