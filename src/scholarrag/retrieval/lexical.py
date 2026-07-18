"""Lexical (keyword) retrieval — Postgres full-text search.

Ranks chunks by how well their words match the query, using the generated
``fts`` tsvector column and its GIN index. This is the BM25-style lexical side
of hybrid retrieval. (Postgres ``ts_rank`` is the built-in lexical ranker — not
literally BM25, which needs an extension, but a solid stand-in; noted in
docs/DESIGN.md.)
"""

from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from scholarrag.db.models import Chunk, Document
from scholarrag.retrieval.base import RetrievedChunk


class LexicalRetriever:
    """Keyword retrieval via Postgres full-text search over ``chunks.fts``."""

    def retrieve(self, session: Session, query: str, *, top_k: int = 10) -> list[RetrievedChunk]:
        "Return the ``top_k`` chunks whose text best matches the query terms."

        tsquery = func.websearch_to_tsquery("english", query)
        rank = func.ts_rank(Chunk.fts, tsquery).label("rank")

        stmt = (
            select(Chunk, rank, Document.filename)
            .join(Document, Chunk.document_id == Document.id)
            .where(Chunk.fts.op("@@")(tsquery))
            .order_by(rank.desc())
            .limit(top_k)
        )

        chunks = session.execute(stmt).all()
        result = []
        for chunk, rank_value, filename in chunks:
            result.append(
                RetrievedChunk(
                    id=chunk.vector_id,
                    document_id=chunk.document_id,
                    chunk_index=chunk.chunk_index,
                    text=chunk.text,
                    filename=filename,
                    score=float(rank_value),
                )
            )

        return result
