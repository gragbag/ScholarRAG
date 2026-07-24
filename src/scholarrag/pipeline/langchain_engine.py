"""The LangChain (LCEL) implementation of the query engine.

A second brain behind the same interface as the hand-rolled
:class:`~scholarrag.pipeline.engine.QueryEngine`, selected via
``PIPELINE=langchain``. Retrieval stays on the hand-rolled
:class:`HybridRetriever` (identical in both pipelines, so the A/B isolates the
orchestration/generation layer); what LangChain replaces is the glue —
prompt -> model -> parser becomes one LCEL chain, and streaming comes free from
``chain.stream``.

Cross-cutting policy is re-attached on this path too: citations map back to
sources, ``enforce_grounding`` gates the result, and the cache wraps ``query``
— the maintenance cost of a second pipeline, made visible on purpose.

LangChain imports stay inside methods (``langchain`` extra, absent in CI).
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

from sqlalchemy.orm import Session

from scholarrag.cache.answer_cache import AnswerCache
from scholarrag.config import Settings
from scholarrag.generation.base import Answer
from scholarrag.generation.citations import extract_citations
from scholarrag.generation.prompts import GROUNDED_SYSTEM, format_sources
from scholarrag.guardrails.output import enforce_grounding
from scholarrag.retrieval.base import RetrievedChunk, Retriever

if TYPE_CHECKING:  # pragma: no cover
    from langchain_core.language_models import BaseChatModel

# The human turn of the chat prompt — LCEL fills {context} and {question}.
_HUMAN_TEMPLATE = "Sources:\n{context}\n\nQuestion: {question}\n\nAnswer with citations:"


def build_langchain_llm(settings: Settings) -> BaseChatModel:
    """The generation model as a LangChain chat model (Gemini strong tier)."""
    from langchain_google_genai import ChatGoogleGenerativeAI

    from scholarrag.observability import get_langchain_callbacks

    return ChatGoogleGenerativeAI(
        model=settings.gemini_model_strong,
        google_api_key=settings.gemini_api_key,
        max_output_tokens=settings.llm_max_output_tokens,
        thinking_budget=0,  # parity with GeminiLLM: thinking disabled
        callbacks=get_langchain_callbacks(settings),  # Langfuse visibility
    )


def build_decider_llm(settings: Settings) -> BaseChatModel:
    """The cheap-tier chat model for agent decisions (grading, query rewriting)."""
    from langchain_google_genai import ChatGoogleGenerativeAI

    from scholarrag.observability import get_langchain_callbacks

    return ChatGoogleGenerativeAI(
        model=settings.gemini_model_cheap,
        google_api_key=settings.gemini_api_key,
        max_output_tokens=256,  # decisions are one word / one line
        thinking_budget=0,
        callbacks=get_langchain_callbacks(settings),
    )


class LangChainQueryEngine:
    """LCEL-composed pipeline: same surface as ``QueryEngine``, different guts."""

    def __init__(
        self,
        *,
        retriever: Retriever,
        llm: BaseChatModel,
        cache: AnswerCache | None = None,
        top_k: int = 5,
    ) -> None:
        self._retriever = retriever
        self._llm = llm
        self._cache = cache
        self._top_k = top_k
        self._chain = self._build_chain()

    def _build_chain(self) -> Any:
        "Compose the LCEL generation chain: prompt | llm | parser."

        from langchain_core.prompts import ChatPromptTemplate

        prompt = ChatPromptTemplate.from_messages(
            [("system", GROUNDED_SYSTEM), ("human", _HUMAN_TEMPLATE)]
        )

        from langchain_core.output_parsers import StrOutputParser

        return prompt | self._llm | StrOutputParser()

    def _retrieve(self, session: Session, query: str) -> list[RetrievedChunk]:
        """Hand-rolled retrieval, identical to the other pipeline (by design)."""
        return self._retriever.retrieve(session, query, top_k=self._top_k)

    def _to_answer(self, text: str, chunks: list[RetrievedChunk]) -> Answer:
        "Chain output (a string) -> gated :class:`Answer`."
        cited = extract_citations(text)

        sources = [chunks[n - 1] for n in cited if 1 <= n <= len(chunks)]

        return enforce_grounding(Answer(text=text, sources=sources))

    # ── public surface (scaffolded — mirrors QueryEngine) ────────────────────

    def query(self, session: Session, query: str) -> Answer:
        """Cache-aside -> retrieve -> LCEL chain -> gated Answer."""
        if self._cache is not None:
            hit = self._cache.get(query)
            if hit is not None:
                return hit

        chunks = self._retrieve(session, query)
        text = self._chain.invoke({"context": format_sources(chunks), "question": query})
        answer = self._to_answer(text, chunks)
        if self._cache is not None:
            self._cache.put(query, answer)
        return answer

    def answer_with_context(
        self, session: Session, query: str
    ) -> tuple[Answer, list[RetrievedChunk]]:
        """Like :meth:`query` but also returns the retrieved contexts (for eval)."""
        chunks = self._retrieve(session, query)
        text = self._chain.invoke({"context": format_sources(chunks), "question": query})
        return self._to_answer(text, chunks), chunks

    def stream(self, session: Session, query: str) -> tuple[list[RetrievedChunk], Iterator[str]]:
        "Streaming variant — LCEL's freebie."

        chunks = self._retrieve(session, query)

        tokens = self._chain.stream({"context": format_sources(chunks), "question": query})

        return chunks, tokens
