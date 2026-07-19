"""QueryEngine tests (Exercise B — the capstone; hermetic).

Uses a real ``QueryRewriter`` + ``Answerer`` (backed by ``FakeLLM``) and a stub
retriever, so no key/Postgres/models. Needs exercises A and C done too. Remove
the ``@pytest.mark.skip`` once ``QueryEngine.query`` works.
"""

from __future__ import annotations

import uuid

from sqlalchemy.orm import Session

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
    """Returns a fixed ranked list, ignoring the query (structurally a Retriever)."""

    def __init__(self, hits: list[RetrievedChunk]) -> None:
        self._hits = hits

    def retrieve(self, session: Session, query: str, *, top_k: int = 10) -> list[RetrievedChunk]:
        return self._hits[:top_k]


# ── Exercise B — QueryEngine.query ───────────────────────────────────────────
def test_query_engine_runs_full_flow() -> None:
    rewriter = QueryRewriter(llm=FakeLLM(["neural networks basics\ndeep learning intro"]))
    retriever = StubRetriever([_chunk("0", "neural nets learn"), _chunk("1", "unrelated")])
    answerer = Answerer(llm=FakeLLM(["Neural nets learn representations [1]."]))
    engine = QueryEngine(
        rewriter=rewriter, retriever=retriever, answerer=answerer, top_k=2, multi_query=True
    )

    answer = engine.query(Session(), "how do neural networks work")

    assert answer.text == "Neural nets learn representations [1]."
    assert answer.sources[0].id == "0"  # the cited, fused top source
