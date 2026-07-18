"""Retriever protocol and the shared result type.

A retriever takes a query string and returns a ranked list of chunks. Two
implementations — :class:`~scholarrag.retrieval.dense.DenseRetriever` (semantic,
via embeddings) and :class:`~scholarrag.retrieval.lexical.LexicalRetriever`
(keyword, via Postgres full-text search) — return the *same* ``RetrievedChunk``
type, so downstream fusion/reranking/generation don't care where a chunk came
from.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from sqlalchemy.orm import Session


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    """One retrieval hit.

    ``id`` is the ``vector_id`` (``"{document_id}:{chunk_index}"``) — the same
    key in the vector store and in Postgres, so fusion (Step 2) can tell when
    both retrievers found the *same* chunk.
    """

    id: str
    document_id: uuid.UUID
    chunk_index: int
    text: str
    filename: str
    score: float  # retriever-specific: cosine (dense) or ts_rank (lexical)


@runtime_checkable
class Retriever(Protocol):
    """Ranked retrieval over the corpus. ``session`` is the per-request DB session.

    Both retrievers accept it (for a uniform call site in the hybrid retriever);
    the dense retriever ignores it, since it reads from the vector store.
    """

    def retrieve(self, session: Session, query: str, *, top_k: int = 10) -> list[RetrievedChunk]:
        """Return up to ``top_k`` chunks most relevant to ``query``, best first."""
        ...
