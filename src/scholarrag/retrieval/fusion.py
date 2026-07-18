"""Reciprocal Rank Fusion — merge ranked lists from multiple retrievers.

Dense (cosine) and lexical (``ts_rank``) produce scores on *incomparable* scales,
so we can't simply add them. RRF sidesteps that by combining **rank positions**
only: a chunk's fused score is the sum, over each retriever that returned it, of
``1 / (k + rank)`` (``rank`` is 1-based). A chunk found by *both* retrievers gets
two contributions and rises to the top — exactly the "both engines independently
agree" signal we want. ``k`` (conventionally 60) damps the influence of the very
top ranks, so no single over-confident retriever can dominate.

The join key across lists is ``RetrievedChunk.id`` (the ``vector_id``): identical
id in two lists means the two retrievers found the *same* chunk.
"""

from __future__ import annotations

from dataclasses import replace

from scholarrag.retrieval.base import RetrievedChunk


def reciprocal_rank_fusion(
    result_lists: list[list[RetrievedChunk]],
    *,
    k: int = 60,
    top_k: int | None = None,
) -> list[RetrievedChunk]:
    "Fuse several ranked lists into one, ordered best-first by RRF score."
    scores: dict[str, float] = {}
    chunk_by_id: dict[str, RetrievedChunk] = {}
    for results in result_lists:
        for rank, chunk in enumerate(results, start=1):
            scores[chunk.id] = scores.get(chunk.id, 0.0) + 1.0 / (k + rank)
            chunk_by_id.setdefault(chunk.id, chunk)

    ordered_ids = sorted(scores, key=lambda cid: scores[cid], reverse=True)
    fused = [replace(chunk_by_id[cid], score=scores[cid]) for cid in ordered_ids]
    return fused if top_k is None else fused[:top_k]
