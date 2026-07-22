"""Pipeline package — the end-to-end RAG query engine.

:class:`QueryEngine` ties rewriting, hybrid retrieval, fusion, and grounded
generation into one call. Use :func:`build_query_engine` to assemble the one
implied by settings (needs an LLM key for the rewriter/answerer).
"""

from __future__ import annotations

from scholarrag.cache import build_answer_cache
from scholarrag.config import Settings, get_settings
from scholarrag.embeddings import build_embedder
from scholarrag.generation import build_answerer
from scholarrag.llm import build_llm_client
from scholarrag.pipeline.engine import QueryEngine
from scholarrag.retrieval import build_hybrid_retriever
from scholarrag.retrieval.rewrite import QueryRewriter

__all__ = [
    "QueryEngine",
    "build_query_engine",
]


def build_query_engine(settings: Settings | None = None) -> QueryEngine:
    """Assemble the full query engine (rewriter + retriever + answerer + cache)."""
    settings = settings or get_settings()
    llm = build_llm_client(settings)  # shared by rewriter and answerer
    embedder = build_embedder(settings)  # shared by retrieval and the semantic cache
    return QueryEngine(
        rewriter=QueryRewriter(llm=llm, num_variations=settings.num_query_variations),
        retriever=build_hybrid_retriever(settings, embedder=embedder),
        answerer=build_answerer(settings, llm=llm),
        cache=build_answer_cache(settings, embedder),  # None when disabled
        top_k=settings.retrieval_top_k,
        multi_query=settings.query_rewriting_enabled,
    )
