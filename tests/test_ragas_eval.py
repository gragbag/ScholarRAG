"""Test the RAGAS-eval plumbing that doesn't need RAGAS.

``collect_samples`` runs the QueryEngine and shapes its output — testable with the
fakes (no eval extra, no LLM). ``build_judge`` / ``run_ragas_eval`` are the Step 2
exercises and need real RAGAS + a key, so they're exercised via ``make eval-rag``.
"""

from __future__ import annotations

import uuid

from sqlalchemy.orm import Session

from scholarrag.eval.dataset import EvalExample
from scholarrag.eval.ragas_eval import collect_samples
from scholarrag.generation import Answerer
from scholarrag.llm import FakeLLM
from scholarrag.pipeline import QueryEngine
from scholarrag.retrieval import RetrievedChunk
from scholarrag.retrieval.rewrite import QueryRewriter


def _chunk(cid: str, text: str) -> RetrievedChunk:
    return RetrievedChunk(
        id=cid, document_id=uuid.uuid4(), chunk_index=0, text=text, filename="a.txt", score=0.0
    )


class StubRetriever:
    def __init__(self, hits: list[RetrievedChunk]) -> None:
        self._hits = hits

    def retrieve(self, session: Session, query: str, *, top_k: int = 10) -> list[RetrievedChunk]:
        return self._hits[:top_k]


def test_collect_samples_gathers_answer_and_contexts() -> None:
    engine = QueryEngine(
        rewriter=QueryRewriter(llm=FakeLLM(["variant one\nvariant two"])),
        retriever=StubRetriever([_chunk("0", "neural nets learn"), _chunk("1", "unrelated")]),
        answerer=Answerer(llm=FakeLLM(["Neural nets learn representations [1]."])),
        top_k=2,
    )
    examples = [
        EvalExample("how do neural networks work", ["a.txt"], "They learn representations.")
    ]

    samples = collect_samples(engine, Session(), examples)

    assert len(samples) == 1
    assert samples[0].question == "how do neural networks work"
    assert samples[0].answer == "Neural nets learn representations [1]."
    assert samples[0].contexts == ["neural nets learn", "unrelated"]  # the retrieved passages
    assert samples[0].reference == "They learn representations."
