"""Hybrid retriever — the full two-stage pipeline behind one ``Retriever``.

Stage 1 (recall): run the dense and lexical retrievers, each returning a large
``candidate_k`` pool, then fuse the two lists with Reciprocal Rank Fusion.
Stage 2 (precision): if a reranker is configured, let it reorder the fused
shortlist; otherwise just take the fused top ``top_k``.

Because ``HybridRetriever`` itself satisfies the :class:`Retriever` protocol,
everything downstream (Step 4's generation) treats it exactly like a plain
retriever — it just happens to be smarter inside.
"""

from __future__ import annotations

from sqlalchemy.orm import Session

from scholarrag.observability import get_tracer
from scholarrag.retrieval.base import RetrievedChunk, Retriever
from scholarrag.retrieval.fusion import reciprocal_rank_fusion
from scholarrag.retrieval.rerank import Reranker


class HybridRetriever:
    """Composes dense + lexical retrieval, RRF fusion, and optional reranking."""

    def __init__(
        self,
        *,
        dense: Retriever,
        lexical: Retriever,
        reranker: Reranker | None = None,
        candidate_k: int = 50,
        rrf_k: int = 60,
    ) -> None:
        self._dense = dense
        self._lexical = lexical
        self._reranker = reranker
        self._candidate_k = candidate_k
        self._rrf_k = rrf_k

    def retrieve(self, session: Session, query: str, *, top_k: int = 10) -> list[RetrievedChunk]:
        "Retrieve, fuse, and (optionally) rerank — best ``top_k`` chunks first."
        tracer = get_tracer("scholarrag.retrieval")

        with tracer.start_as_current_span("retrieve.dense"):
            dense_hits = self._dense.retrieve(session, query, top_k=self._candidate_k)

        with tracer.start_as_current_span("retrieve.lexical"):
            lexical_hits = self._lexical.retrieve(session, query, top_k=self._candidate_k)

        fused = reciprocal_rank_fusion(
            [dense_hits, lexical_hits], k=self._rrf_k, top_k=self._candidate_k
        )

        if self._reranker is None:
            return fused[:top_k]

        return self._reranker.rerank(query, fused, top_k=top_k)
