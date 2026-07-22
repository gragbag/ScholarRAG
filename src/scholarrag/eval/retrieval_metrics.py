"""Hand-rolled information-retrieval metrics (document-level, binary relevance).

Each takes a ranked list of *distinct* filenames (best first) and the set of
relevant filenames for the query. ``precision_at_k`` is implemented as your
template; ``recall_at_k`` / ``reciprocal_rank`` / ``ndcg_at_k`` are the exercises.
"""

from __future__ import annotations

import math


def precision_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    """Fraction of the top-k retrieved docs that are relevant. (Worked example.)"""
    if k <= 0:
        return 0.0
    top = ranked[:k]
    return sum(1 for f in top if f in relevant) / k


def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    "Fraction of the relevant docs that appear in the top-k."

    if not relevant:
        return 0.0

    hits = len(set(ranked[:k]) & relevant)

    return hits / len(relevant)


def reciprocal_rank(ranked: list[str], relevant: set[str]) -> float:
    "1 / (rank of the first relevant doc); 0 if none. Averaged over queries = MRR."

    for position, filename in enumerate(ranked, start=1):
        if filename in relevant:
            return 1.0 / position

    return 0.0


def ndcg_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    "Normalized Discounted Cumulative Gain at k — ranking quality (binary gain)."

    dcg = sum(1.0 / math.log2(i + 1) for i, f in enumerate(ranked[:k], start=1) if f in relevant)

    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))

    return dcg / idcg if idcg > 0 else 0.0
