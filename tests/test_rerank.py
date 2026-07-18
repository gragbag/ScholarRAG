"""Reranker tests.

``FakeReranker`` and the protocol test pass now. The cross-encoder test is
Exercise C — remove its ``@pytest.mark.skip`` once you've implemented
``CrossEncoderReranker.rerank`` (it injects a stub ``predict_fn``, so no torch).
"""

from __future__ import annotations

import uuid

import pytest

from scholarrag.retrieval import Reranker, RetrievedChunk
from scholarrag.retrieval.rerank import CrossEncoderReranker, FakeReranker


def _chunk(cid: str, text: str) -> RetrievedChunk:
    return RetrievedChunk(
        id=cid, document_id=uuid.uuid4(), chunk_index=0, text=text, filename="a.txt", score=0.0
    )


def test_rerankers_conform_to_protocol() -> None:
    assert isinstance(FakeReranker(), Reranker)
    assert isinstance(CrossEncoderReranker(predict_fn=lambda pairs: [0.0]), Reranker)


def test_fake_reranker_orders_by_query_overlap() -> None:
    chunks = [
        _chunk("0", "a shortbread recipe with butter and sugar"),
        _chunk("1", "neural networks learn representations with attention"),
    ]
    result = FakeReranker().rerank("how do neural networks learn", chunks, top_k=2)
    assert result[0].id == "1"  # more query words overlap the neural chunk


# ── Exercise C — CrossEncoderReranker.rerank ─────────────────────────────────
def test_cross_encoder_rerank_orders_by_score() -> None:
    chunks = [_chunk("0", "first"), _chunk("1", "second"), _chunk("2", "third")]
    # Stub the model: score the 2nd pair highest, then the 3rd, then the 1st.
    reranker = CrossEncoderReranker(predict_fn=lambda pairs: [0.1, 0.9, 0.5])
    result = reranker.rerank("q", chunks, top_k=2)
    assert [c.id for c in result] == ["1", "2"]
    assert result[0].score == pytest.approx(0.9)
