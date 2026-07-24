"""LangChain-engine tests — hermetic via LangChain's own fake chat model.

The whole module needs ``langchain_core`` (the ``langchain`` extra), so it skips
in CI. Locally, the three exercise targets unskip as you implement A/B/C.
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy.orm import Session

pytest.importorskip("langchain_core")  # langchain extra — absent in CI

from scholarrag.retrieval import RetrievedChunk


def _chunk(cid: str, text: str) -> RetrievedChunk:
    return RetrievedChunk(
        id=cid, document_id=uuid.uuid4(), chunk_index=0, text=text, filename="a.txt", score=0.5
    )


class StubRetriever:
    def retrieve(self, session: Session, query: str, *, top_k: int = 10) -> list[RetrievedChunk]:
        return [_chunk("0", "RAG grounds answers"), _chunk("1", "unrelated")][:top_k]


def _engine(responses: list[str]) -> object:
    from langchain_core.language_models import FakeListChatModel

    from scholarrag.pipeline import LangChainQueryEngine

    return LangChainQueryEngine(
        retriever=StubRetriever(),
        llm=FakeListChatModel(responses=responses),
        top_k=2,
    )


# ── Exercise A — the LCEL chain ──────────────────────────────────────────────
def test_chain_composes_and_generates() -> None:
    engine = _engine(["RAG grounds answers [1]."])
    answer, chunks = engine.answer_with_context(Session(), "how does RAG work")  # type: ignore[attr-defined]
    assert answer.text == "RAG grounds answers [1]."
    assert len(chunks) == 2  # the stubbed retrieval, unchanged by the chain


# ── Exercise B — citations + grounding gate re-attached ──────────────────────
def test_to_answer_maps_citations_and_gates() -> None:
    from scholarrag.guardrails import REFUSAL_MESSAGE

    # Cited -> sources mapped (needs exercise A too, for the chain).
    engine = _engine(["RAG grounds answers [1]."])
    answer = engine.query(Session(), "how does RAG work")  # type: ignore[attr-defined]
    assert [s.id for s in answer.sources] == ["0"]

    # Uncited assertion -> the grounding gate replaces it, same as the other pipeline.
    engine2 = _engine(["Everything is fine, trust me."])
    gated = engine2.query(Session(), "how does RAG work")  # type: ignore[attr-defined]
    assert gated.text == REFUSAL_MESSAGE


# ── Exercise C — streaming + the toggle ──────────────────────────────────────
def test_stream_yields_deltas() -> None:
    engine = _engine(["RAG grounds answers [1]."])
    chunks, tokens = engine.stream(Session(), "how does RAG work")  # type: ignore[attr-defined]
    assert len(chunks) == 2
    assert "".join(tokens) == "RAG grounds answers [1]."


def test_toggle_builds_langchain_engine() -> None:
    from scholarrag.config import Settings
    from scholarrag.pipeline import LangChainQueryEngine, build_query_engine

    settings = Settings(
        _env_file=None,
        pipeline="langchain",
        gemini_api_key="fake-key",  # construction only; no network
        embedding_provider="fake",
        vector_store="local",
    )
    engine = build_query_engine(settings)
    assert isinstance(engine, LangChainQueryEngine)
