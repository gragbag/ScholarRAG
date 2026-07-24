"""Comparison-harness tests — hermetic.

``load_hard_examples`` and ``is_grounded_answer`` pass now; the two skipped
tests are the Step 2 exercise targets. Skips in CI (langchain extra absent).
"""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest

pytest.importorskip("langchain_core")  # langchain extra — absent in CI

from scholarrag.eval.compare import (
    CountingCallback,
    QuestionResult,
    is_grounded_answer,
    load_hard_examples,
    score_results,
)
from scholarrag.generation.base import Answer
from scholarrag.guardrails import REFUSAL_MESSAGE
from scholarrag.retrieval import RetrievedChunk

HARD_SET = Path(__file__).resolve().parents[1] / "data" / "eval" / "hard.json"


def _chunk(filename: str = "a.txt") -> RetrievedChunk:
    return RetrievedChunk(
        id="d:0", document_id=uuid.uuid4(), chunk_index=0, text="t", filename=filename, score=0.5
    )


def _result(
    *,
    answerable: bool,
    answered: bool,
    cited: list[str],
    relevant: list[str],
    calls: int = 2,
    latency: float = 1.0,
) -> QuestionResult:
    return QuestionResult(
        question="q",
        answerable=answerable,
        relevant_files=relevant,
        answered=answered,
        cited_files=cited,
        llm_calls=calls,
        latency_s=latency,
    )


# ── pass now ─────────────────────────────────────────────────────────────────
def test_hard_set_loads_with_controls() -> None:
    examples = load_hard_examples(HARD_SET)
    assert len(examples) == 11
    assert sum(not e.answerable for e in examples) == 3  # the refusal controls
    assert all(e.relevant_files == [] for e in examples if not e.answerable)


def test_is_grounded_answer_distinguishes_refusals() -> None:
    assert is_grounded_answer(Answer(text="cited claim [1]", sources=[_chunk()])) is True
    assert is_grounded_answer(Answer(text=REFUSAL_MESSAGE, sources=[])) is False
    assert (
        is_grounded_answer(
            Answer(text="The sources do not contain enough information.", sources=[])
        )
        is False
    )


# ── Exercise A — the scoring function ────────────────────────────────────────
def test_score_results_separates_populations() -> None:
    results = [
        # answerable, answered, cited the right file -> counts for H1 and source-hit
        _result(answerable=True, answered=True, cited=["rag.md"], relevant=["rag.md"], calls=2),
        # answerable but refused -> H1 miss AND source-hit miss
        _result(answerable=True, answered=False, cited=[], relevant=["a.pdf"], calls=4),
        # UNanswerable but answered -> the H2 failure case
        _result(answerable=False, answered=True, cited=["b.pdf"], relevant=[], calls=6),
    ]
    report = score_results(results)

    assert report.answered_rate == pytest.approx(1 / 2)  # over answerable only
    assert report.false_answer_rate == pytest.approx(1.0)  # over unanswerable only
    assert report.source_hit_rate == pytest.approx(1 / 2)
    assert report.mean_llm_calls == pytest.approx(4.0)  # over everything
    assert report.n_answerable == 2
    assert report.n_unanswerable == 1


# ── Exercise B — the counting callback ───────────────────────────────────────
def test_counting_callback_counts_model_calls() -> None:
    from langchain_core.language_models import FakeListChatModel
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate

    counter = CountingCallback()
    model = FakeListChatModel(responses=["hi"], callbacks=[counter])
    chain = ChatPromptTemplate.from_messages([("human", "{q}")]) | model | StrOutputParser()

    chain.invoke({"q": "one"})
    chain.invoke({"q": "two"})

    assert counter.count == 2  # one per model invocation, none swallowed
