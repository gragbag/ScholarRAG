"""Pipeline package — the end-to-end RAG query engine.

Two interchangeable implementations behind one surface, selected by
``PIPELINE``: the hand-rolled :class:`QueryEngine` (Phase 2 baseline) and the
LCEL-composed :class:`LangChainQueryEngine` (post-MVP rewrite). Use
:func:`build_query_engine` to assemble whichever settings imply — the API,
eval harness, and tests don't know or care which brain is running.
"""

from __future__ import annotations

from scholarrag.cache import build_answer_cache
from scholarrag.config import Settings, get_settings
from scholarrag.embeddings import build_embedder
from scholarrag.generation import build_answerer
from scholarrag.llm import build_llm_client
from scholarrag.pipeline.agentic_engine import AgenticQueryEngine
from scholarrag.pipeline.engine import QueryEngine
from scholarrag.pipeline.langchain_engine import LangChainQueryEngine
from scholarrag.retrieval import build_hybrid_retriever
from scholarrag.retrieval.rewrite import QueryRewriter

__all__ = [
    "AgenticQueryEngine",
    "LangChainQueryEngine",
    "QueryEngine",
    "build_query_engine",
]

# The engines share a surface (query / answer_with_context / stream), not a base
# class — same structural-typing approach as Retriever/LLMClient.
AnyQueryEngine = QueryEngine | LangChainQueryEngine | AgenticQueryEngine


def build_query_engine(settings: Settings | None = None) -> AnyQueryEngine:
    """Assemble the query engine implied by ``PIPELINE`` (retriever + cache shared)."""
    settings = settings or get_settings()
    embedder = build_embedder(settings)  # shared by retrieval and the semantic cache
    retriever = build_hybrid_retriever(settings, embedder=embedder)
    cache = build_answer_cache(settings, embedder)  # None when disabled

    if settings.pipeline == "langchain":
        from scholarrag.pipeline.langchain_engine import build_langchain_llm

        return LangChainQueryEngine(
            retriever=retriever,
            llm=build_langchain_llm(settings),
            cache=cache,
            top_k=settings.retrieval_top_k,
        )

    if settings.pipeline == "agentic":
        from scholarrag.pipeline.langchain_engine import build_decider_llm, build_langchain_llm

        return AgenticQueryEngine(
            retriever=retriever,
            llm=build_langchain_llm(settings),  # strong tier: final generation
            decider_llm=build_decider_llm(settings),  # cheap tier: grade + rewrite
            cache=cache,
            top_k=settings.retrieval_top_k,
            max_iterations=settings.max_agent_iterations,
        )

    llm = build_llm_client(settings)  # shared by rewriter and answerer
    return QueryEngine(
        rewriter=QueryRewriter(llm=llm, num_variations=settings.num_query_variations),
        retriever=retriever,
        answerer=build_answerer(settings, llm=llm),
        cache=cache,
        top_k=settings.retrieval_top_k,
        multi_query=settings.query_rewriting_enabled,
    )
