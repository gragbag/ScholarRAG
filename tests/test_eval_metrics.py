"""Retrieval-metric unit tests.

``precision_at_k`` (the worked example) passes now. The other three are the
Step 1 exercises — remove each ``@pytest.mark.skip`` once implemented.
"""

from __future__ import annotations

import pytest

from scholarrag.eval import ndcg_at_k, precision_at_k, recall_at_k, reciprocal_rank

# A ranked list of distinct filenames (best first) and the relevant set.
RANKED = ["rag.md", "embeddings.md", "transformers.txt", "other.md"]
RELEVANT = {"transformers.txt", "rag.md"}  # relevant docs sit at ranks 1 and 3


def test_precision_at_k() -> None:
    assert precision_at_k(RANKED, RELEVANT, 3) == pytest.approx(2 / 3)  # 2 of top-3 relevant
    assert precision_at_k(RANKED, RELEVANT, 1) == pytest.approx(1.0)  # rank 1 is relevant
    assert precision_at_k(RANKED, set(), 3) == 0.0  # nothing relevant


# ── Exercise A — recall_at_k ─────────────────────────────────────────────────
def test_recall_at_k() -> None:
    assert recall_at_k(RANKED, RELEVANT, 3) == pytest.approx(1.0)  # both relevant in top-3
    assert recall_at_k(RANKED, RELEVANT, 1) == pytest.approx(0.5)  # only 1 of 2 in top-1
    assert recall_at_k(RANKED, set(), 3) == 0.0  # undefined -> 0


# ── Exercise B — reciprocal_rank ─────────────────────────────────────────────
def test_reciprocal_rank() -> None:
    assert reciprocal_rank(RANKED, RELEVANT) == pytest.approx(1.0)  # first relevant at rank 1
    assert reciprocal_rank(RANKED, {"transformers.txt"}) == pytest.approx(1 / 3)  # rank 3
    assert reciprocal_rank(RANKED, {"missing.md"}) == 0.0  # none found


# ── Exercise C — ndcg_at_k ───────────────────────────────────────────────────
def test_ndcg_at_k() -> None:
    import math

    # Relevant at ranks 1 and 3. DCG = 1/log2(2) + 1/log2(4) = 1.0 + 0.5 = 1.5.
    # IDCG (both at top) = 1/log2(2) + 1/log2(3) = 1.0 + 0.6309 = 1.6309.
    expected = (1.0 + 0.5) / (1.0 + 1.0 / math.log2(3))
    assert ndcg_at_k(RANKED, RELEVANT, 4) == pytest.approx(expected)
    assert ndcg_at_k(RANKED, set(), 4) == 0.0  # nothing relevant -> 0
    # A perfect ranking scores 1.0.
    assert ndcg_at_k(["a", "b", "c"], {"a"}, 3) == pytest.approx(1.0)
