"""Retrieval — dense (semantic) and lexical (keyword) search, fused and reranked.

Stage 1 builds two complementary engines — :class:`DenseRetriever` (embeddings)
and :class:`LexicalRetriever` (Postgres FTS) — that return the same
:class:`RetrievedChunk` type. Step 2 combines them: :func:`reciprocal_rank_fusion`
merges their ranked lists, an optional :class:`Reranker` sharpens the shortlist,
and :class:`HybridRetriever` wires the whole pipeline behind the ``Retriever``
protocol. Use :func:`build_hybrid_retriever` to assemble the one implied by
settings.
"""

from __future__ import annotations

from scholarrag.config import Settings, get_settings
from scholarrag.embeddings import Embedder, build_embedder
from scholarrag.retrieval.base import RetrievedChunk, Retriever
from scholarrag.retrieval.dense import DenseRetriever
from scholarrag.retrieval.fusion import reciprocal_rank_fusion
from scholarrag.retrieval.hybrid import HybridRetriever
from scholarrag.retrieval.lexical import LexicalRetriever
from scholarrag.retrieval.rerank import CrossEncoderReranker, FakeReranker, Reranker
from scholarrag.vectorstore import VectorStore, build_vector_store

__all__ = [
    "CrossEncoderReranker",
    "DenseRetriever",
    "FakeReranker",
    "HybridRetriever",
    "LexicalRetriever",
    "Reranker",
    "RetrievedChunk",
    "Retriever",
    "build_hybrid_retriever",
    "build_reranker",
    "reciprocal_rank_fusion",
]


def build_reranker(settings: Settings | None = None) -> Reranker | None:
    """Return the reranker implied by ``RERANKER_PROVIDER`` (``None`` = fusion only)."""
    settings = settings or get_settings()
    provider = settings.reranker_provider
    if provider == "none":
        return None
    if provider == "fake":
        return FakeReranker()
    if provider == "cross_encoder":
        return CrossEncoderReranker(settings.reranker_model)
    raise ValueError(f"unsupported reranker provider: {provider!r}")


def build_hybrid_retriever(
    settings: Settings | None = None,
    *,
    embedder: Embedder | None = None,
    vector_store: VectorStore | None = None,
) -> HybridRetriever:
    """Assemble the full hybrid retriever (dense + lexical + fusion + rerank).

    ``embedder`` / ``vector_store`` can be injected (tests wire in the fakes);
    otherwise they're built from settings.
    """
    settings = settings or get_settings()
    embedder = embedder or build_embedder(settings)
    vector_store = vector_store or build_vector_store(settings)
    return HybridRetriever(
        dense=DenseRetriever(embedder=embedder, vector_store=vector_store),
        lexical=LexicalRetriever(),
        reranker=build_reranker(settings),
        candidate_k=settings.retrieval_candidate_k,
        rrf_k=settings.rrf_k,
    )
