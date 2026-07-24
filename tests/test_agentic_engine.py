"""Agentic-engine tests — hermetic; they assert the agent's *decisions*.

Scripted fake chat models make grading/rewriting/generation deterministic, and a
recording stub retriever exposes the trajectory (how many retrievals, with which
queries). The toggle test passes now; the rest are the Step 1 exercise targets.
Skips in CI (``agentic`` extra absent).
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy.orm import Session

pytest.importorskip("langgraph")  # agentic extra — absent in CI

from scholarrag.pipeline.agentic_engine import AgenticQueryEngine
from scholarrag.retrieval import RetrievedChunk


def _chunk(cid: str, text: str) -> RetrievedChunk:
    return RetrievedChunk(
        id=cid, document_id=uuid.uuid4(), chunk_index=0, text=text, filename="a.txt", score=0.5
    )


class RecordingRetriever:
    """Returns fixed chunks; records every query so tests can assert trajectories."""

    def __init__(self) -> None:
        self.queries: list[str] = []

    def retrieve(self, session: Session, query: str, *, top_k: int = 10) -> list[RetrievedChunk]:
        self.queries.append(query)
        return [_chunk("0", "RAG grounds answers"), _chunk("1", "unrelated")][:top_k]


def _engine(
    *, decider: list[str], generator: list[str], max_iterations: int = 2
) -> tuple[AgenticQueryEngine, RecordingRetriever]:
    from langchain_core.language_models import FakeListChatModel

    retriever = RecordingRetriever()
    engine = AgenticQueryEngine(
        retriever=retriever,
        llm=FakeListChatModel(responses=generator),
        decider_llm=FakeListChatModel(responses=decider),
        top_k=2,
        max_iterations=max_iterations,
    )
    return engine, retriever


def test_toggle_builds_agentic_engine() -> None:
    from scholarrag.config import Settings
    from scholarrag.pipeline import build_query_engine

    settings = Settings(
        _env_file=None,
        pipeline="agentic",
        gemini_api_key="fake-key",  # construction only; the graph is built lazily
        embedding_provider="fake",
        vector_store="local",
    )
    assert isinstance(build_query_engine(settings), AgenticQueryEngine)


# ── Exercise A — the graph (happy path) ──────────────────────────────────────
def test_happy_path_is_two_llm_calls() -> None:
    # Grader says relevant immediately; generator cites -> straight through.
    engine, retriever = _engine(decider=["relevant"], generator=["RAG grounds answers [1]."])
    answer = engine.query(Session(), "how does RAG work")

    assert answer.text == "RAG grounds answers [1]."
    assert [s.id for s in answer.sources] == ["0"]
    assert retriever.queries == ["how does RAG work"]  # exactly one retrieval, no loop tax


# ── Exercise B — grading + the rewrite loop ──────────────────────────────────
def test_weak_retrieval_triggers_rewrite_loop() -> None:
    # Decider script: grade #1 -> "weak", rewrite -> new query, grade #2 -> "relevant".
    engine, retriever = _engine(
        decider=["weak", "retrieval augmented generation grounding", "relevant"],
        generator=["RAG grounds answers [1]."],
    )
    answer = engine.query(Session(), "how does RAG work")

    assert answer.text == "RAG grounds answers [1]."
    assert retriever.queries == [
        "how does RAG work",
        "retrieval augmented generation grounding",  # the rewritten retry
    ]


def test_always_weak_stops_at_the_cap_and_answers_best_effort() -> None:
    # Grader never satisfied; rewrites burn the budget; then best-effort generate.
    engine, retriever = _engine(
        decider=["weak", "rewrite one", "weak", "rewrite two", "weak"],
        generator=["RAG grounds answers [1]."],
        max_iterations=2,
    )
    answer = engine.query(Session(), "how does RAG work")

    assert len(retriever.queries) == 3  # original + exactly max_iterations rewrites
    assert answer.text == "RAG grounds answers [1]."  # still answered (grounded)


# ── Exercise C — self-correction after generation ────────────────────────────
def test_uncited_answer_retries_then_gate_refuses() -> None:
    from scholarrag.guardrails import REFUSAL_MESSAGE

    # Both generations come back uncited -> one full retry (budget = 1), then
    # the gate refuses. (With budget 2 the agent would correctly retry twice.)
    engine, retriever = _engine(
        decider=["relevant", "another phrasing", "relevant"],
        generator=["Everything is fine, trust me.", "Still no citations, honestly."],
        max_iterations=1,
    )
    answer = engine.query(Session(), "how does RAG work")

    assert answer.text == REFUSAL_MESSAGE
    assert answer.sources == []
    assert len(retriever.queries) == 2  # the uncited answer bought one more loop


def test_honest_refusal_ends_without_retry() -> None:
    # The model admitting "not covered" is a valid terminal state — no loop.
    engine, retriever = _engine(
        decider=["relevant"],
        generator=["The sources do not contain enough information to answer."],
    )
    answer = engine.query(Session(), "what is the meaning of life")

    assert "do not contain" in answer.text  # preserved, not replaced
    assert retriever.queries == ["what is the meaning of life"]  # no retry spent
