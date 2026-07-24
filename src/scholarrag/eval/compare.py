"""Pipeline comparison on the hard set — deterministic agentic-vs-single-shot metrics.

No LLM judge here (that's deliberate — see BENCHMARKS): the hypotheses are
binary. Did the pipeline produce a grounded answer or refuse? Did it cite the
right file? How many model calls did it spend? Exact counting answers all three.

Not exported from ``scholarrag.eval`` — this module needs langchain_core (the
callback base class), which CI's core-only environment lacks.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from sqlalchemy.orm import Session

from scholarrag.config import Settings
from scholarrag.generation.base import Answer
from scholarrag.guardrails.output import REFUSAL_MESSAGE, looks_like_refusal


@dataclass(frozen=True, slots=True)
class HardExample:
    """One hard-set question; ``answerable=False`` marks a refusal-control."""

    question: str
    relevant_files: list[str]
    answerable: bool
    reference_answer: str | None = None


def load_hard_examples(path: Path) -> list[HardExample]:
    """Load ``data/eval/hard.json`` (golden-style schema + the answerable flag)."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return [
        HardExample(
            question=item["question"],
            relevant_files=list(item["relevant_files"]),
            answerable=bool(item["answerable"]),
            reference_answer=item.get("reference_answer"),
        )
        for item in data
    ]


@dataclass(frozen=True, slots=True)
class QuestionResult:
    """What one pipeline did with one question."""

    question: str
    answerable: bool
    relevant_files: list[str]
    answered: bool  # produced a grounded answer (vs any form of refusal)
    cited_files: list[str]
    llm_calls: int
    latency_s: float


@dataclass(frozen=True, slots=True)
class ComparisonReport:
    """Aggregate metrics for one pipeline over the hard set."""

    answered_rate: float  # grounded answers / answerable questions  (H1: recovery)
    false_answer_rate: float  # answers / UNanswerable questions     (H2: must stay 0)
    source_hit_rate: float  # cited an expected file / answerable    (right evidence?)
    mean_llm_calls: float  # per question                            (H3: the price)
    mean_latency_s: float
    n_answerable: int
    n_unanswerable: int


def is_grounded_answer(answer: Answer) -> bool:
    """True when the pipeline actually answered (vs refusing in any form)."""
    if answer.text == REFUSAL_MESSAGE or looks_like_refusal(answer.text):
        return False
    return bool(answer.sources)


class CountingCallback(BaseCallbackHandler):  # type: ignore[misc]  # base is Any (skipped pkg)
    "Counts model invocations via LangChain's callback mechanism."

    def __init__(self) -> None:
        super().__init__()
        self.count = 0

    def on_chat_model_start(self, serialized: Any, messages: Any, **kwargs: Any) -> None:
        self.count += 1


def collect_results(
    engine: Any,
    session: Session,
    examples: list[HardExample],
    counter: CountingCallback | None = None,
) -> list[QuestionResult]:
    """Run every hard question through ``engine``, recording what happened."""
    results: list[QuestionResult] = []
    for example in examples:
        calls_before = counter.count if counter else 0
        start = time.perf_counter()
        answer = engine.query(session, example.question)
        latency = time.perf_counter() - start
        results.append(
            QuestionResult(
                question=example.question,
                answerable=example.answerable,
                relevant_files=example.relevant_files,
                answered=is_grounded_answer(answer),
                cited_files=sorted({c.filename for c in answer.sources}),
                llm_calls=(counter.count - calls_before) if counter else 0,
                latency_s=latency,
            )
        )
    return results


def score_results(results: list[QuestionResult]) -> ComparisonReport:
    "Aggregate per-question records into the comparison metrics."
    answerable = [r for r in results if r.answerable]
    unanswerable = [r for r in results if not r.answerable]

    def _rate(hits: float, total: int) -> float:
        return hits / total if total else 0.0

    answered_rate = _rate(sum(r.answered for r in answerable), len(answerable))

    false_answer_rate = _rate(sum(r.answered for r in unanswerable), len(unanswerable))

    hits = sum(1 for r in answerable if r.answered and set(r.cited_files) & set(r.relevant_files))
    source_hit_rate = _rate(hits, len(answerable))

    mean_llm_calls = _rate(sum(r.llm_calls for r in results), len(results))
    mean_latency_s = _rate(sum(r.latency_s for r in results), len(results))

    return ComparisonReport(
        answered_rate,
        false_answer_rate,
        source_hit_rate,
        mean_llm_calls,
        mean_latency_s,
        len(answerable),
        len(unanswerable),
    )


def build_engine_with_counter(settings: Settings) -> tuple[Any, CountingCallback]:
    """Build the pipeline implied by ``settings.pipeline`` with a call counter attached.

    Mirrors ``build_query_engine`` but reaches into the chat models to append the
    counter to their callbacks — possible here because the comparison script owns
    engine construction. Cache is expected to be disabled by the caller (a cached
    answer would let one pipeline serve the other's results).
    """
    from scholarrag.cache import build_answer_cache
    from scholarrag.embeddings import build_embedder
    from scholarrag.pipeline import AgenticQueryEngine, LangChainQueryEngine
    from scholarrag.pipeline.langchain_engine import build_decider_llm, build_langchain_llm
    from scholarrag.retrieval import build_hybrid_retriever

    counter = CountingCallback()
    embedder = build_embedder(settings)
    retriever = build_hybrid_retriever(settings, embedder=embedder)
    cache = build_answer_cache(settings, embedder)

    llm = build_langchain_llm(settings)
    llm.callbacks = [*list(llm.callbacks or []), counter]

    if settings.pipeline == "agentic":
        decider = build_decider_llm(settings)
        decider.callbacks = [*list(decider.callbacks or []), counter]
        return (
            AgenticQueryEngine(
                retriever=retriever,
                llm=llm,
                decider_llm=decider,
                cache=cache,
                top_k=settings.retrieval_top_k,
                max_iterations=settings.max_agent_iterations,
            ),
            counter,
        )

    return (
        LangChainQueryEngine(
            retriever=retriever, llm=llm, cache=cache, top_k=settings.retrieval_top_k
        ),
        counter,
    )
