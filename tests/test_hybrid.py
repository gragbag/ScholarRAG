"""HybridRetriever tests (Exercise B).

The protocol-conformance test passes now. The behaviour tests are Exercise B —
they use *stub* dense/lexical retrievers (canned lists), so no Postgres and no
torch. Remove the ``@pytest.mark.skip`` once ``HybridRetriever.retrieve`` works.
"""

from __future__ import annotations

import uuid

from sqlalchemy.orm import Session

from scholarrag.retrieval import FakeReranker, HybridRetriever, RetrievedChunk, Retriever


def _chunk(cid: str, text: str = "") -> RetrievedChunk:
    return RetrievedChunk(
        id=cid,
        document_id=uuid.uuid4(),
        chunk_index=0,
        text=text or cid,
        filename="a.txt",
        score=0.0,
    )


class StubRetriever:
    """A retriever that returns a fixed list, ignoring the query."""

    def __init__(self, hits: list[RetrievedChunk]) -> None:
        self._hits = hits

    def retrieve(self, session: Session, query: str, *, top_k: int = 10) -> list[RetrievedChunk]:
        return self._hits[:top_k]


def test_hybrid_conforms_to_protocol() -> None:
    hybrid = HybridRetriever(dense=StubRetriever([]), lexical=StubRetriever([]))
    assert isinstance(hybrid, Retriever)


# ── Exercise B — HybridRetriever.retrieve ────────────────────────────────────
def test_hybrid_fuses_dense_and_lexical() -> None:
    # "B" is top of both stubs -> RRF must float it to #1.
    dense = StubRetriever([_chunk("B"), _chunk("A"), _chunk("C")])
    lexical = StubRetriever([_chunk("B"), _chunk("D")])
    hybrid = HybridRetriever(dense=dense, lexical=lexical)
    results = hybrid.retrieve(Session(), "anything", top_k=3)
    assert results[0].id == "B"
    assert len(results) == 3


def test_hybrid_applies_reranker_when_present() -> None:
    # The reranker reorders by query-word overlap, overriding fusion order.
    dense = StubRetriever([_chunk("0", "butter and sugar"), _chunk("1", "neural networks")])
    lexical = StubRetriever([])
    hybrid = HybridRetriever(dense=dense, lexical=lexical, reranker=FakeReranker())
    results = hybrid.retrieve(Session(), "neural networks", top_k=2)
    assert results[0].id == "1"
