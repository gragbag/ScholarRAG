"""Reciprocal Rank Fusion tests (Exercise A — hermetic, no Postgres/torch).

Remove ``pytestmark`` once you've implemented ``reciprocal_rank_fusion``.
"""

from __future__ import annotations

import uuid

import pytest

from scholarrag.retrieval import RetrievedChunk, reciprocal_rank_fusion


def _chunk(cid: str) -> RetrievedChunk:
    return RetrievedChunk(
        id=cid, document_id=uuid.uuid4(), chunk_index=0, text=cid, filename="a.txt", score=0.0
    )


def test_rrf_promotes_the_chunk_both_retrievers_rank_highly() -> None:
    # "B" is #1 in both lists; nothing else appears in both -> B must win.
    dense = [_chunk("B"), _chunk("A"), _chunk("C")]
    lexical = [_chunk("B"), _chunk("D"), _chunk("A")]
    fused = reciprocal_rank_fusion([dense, lexical])
    assert fused[0].id == "B"
    # "A" (rank 2 + rank 3) outranks "C"/"D" (each in only one list).
    ids = [c.id for c in fused]
    assert ids.index("A") < ids.index("C")
    assert ids.index("A") < ids.index("D")


def test_rrf_score_is_sum_of_reciprocal_ranks() -> None:
    # One chunk, top of both lists, k=60 -> 1/61 + 1/61.
    fused = reciprocal_rank_fusion([[_chunk("X")], [_chunk("X")]], k=60)
    assert len(fused) == 1
    assert fused[0].score == pytest.approx(2.0 / 61.0)


def test_rrf_respects_top_k() -> None:
    dense = [_chunk("A"), _chunk("B"), _chunk("C")]
    fused = reciprocal_rank_fusion([dense], top_k=2)
    assert [c.id for c in fused] == ["A", "B"]
