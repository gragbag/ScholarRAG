"""The query engine — the whole RAG flow behind one method.

Composes everything you've built: rewrite the query into variations (Step 3),
retrieve for each with the hybrid retriever (Steps 1-2), fuse the result lists
with RRF (Step 2, now *across query phrasings*), and generate a grounded, cited
answer (Step 4). ``QueryEngine.query`` is the public entry point the API calls.
"""

from __future__ import annotations

from collections.abc import Iterator

from sqlalchemy.orm import Session

from scholarrag.generation.answerer import Answerer
from scholarrag.generation.base import Answer
from scholarrag.retrieval.base import RetrievedChunk, Retriever
from scholarrag.retrieval.fusion import reciprocal_rank_fusion
from scholarrag.retrieval.rewrite import QueryRewriter


class QueryEngine:
    """Orchestrates rewrite -> multi-query retrieve -> fuse -> generate."""

    def __init__(
        self,
        *,
        rewriter: QueryRewriter,
        retriever: Retriever,
        answerer: Answerer,
        top_k: int = 5,
        multi_query: bool = True,
    ) -> None:
        self._rewriter = rewriter
        self._retriever = retriever
        self._answerer = answerer
        self._top_k = top_k
        self._multi_query = multi_query

    def _retrieve(self, session: Session, query: str) -> list[RetrievedChunk]:
        """Rewrite -> retrieve per query -> fuse across queries (the retrieval half)."""
        queries = self._rewriter.rewrite(query) if self._multi_query else [query]
        result_lists = [self._retriever.retrieve(session, q, top_k=self._top_k) for q in queries]
        return reciprocal_rank_fusion(result_lists, top_k=self._top_k)

    def query(self, session: Session, query: str) -> Answer:
        "Run the full RAG pipeline for ``query`` and return a grounded answer."
        fused = self._retrieve(session, query)
        return self._answerer.answer(query, fused)

    def stream(self, session: Session, query: str) -> tuple[list[RetrievedChunk], Iterator[str]]:
        """Streaming variant: retrieve synchronously, then stream the answer tokens.

        Returns the fused candidate chunks (so the caller can resolve citations
        once the stream finishes) plus an iterator of answer text deltas.
        """
        fused = self._retrieve(session, query)
        return fused, self._answerer.answer_stream(query, fused)
