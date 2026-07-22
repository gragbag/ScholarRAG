"""Evaluation harness — measure retrieval quality with real IR metrics.

Step 1 is retrieval-only and LLM-free: a filename-labelled dataset
(:class:`EvalExample`), hand-rolled metrics (Recall@k, Precision@k, MRR, nDCG@k),
and :func:`evaluate_retrieval` to score any ``Retriever`` over the dataset.
Generation eval (RAGAS) arrives in Step 2.
"""

from __future__ import annotations

from scholarrag.eval.dataset import EvalExample, load_examples
from scholarrag.eval.retrieval_metrics import (
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from scholarrag.eval.runner import RetrievalReport, evaluate_retrieval

__all__ = [
    "EvalExample",
    "RetrievalReport",
    "evaluate_retrieval",
    "load_examples",
    "ndcg_at_k",
    "precision_at_k",
    "recall_at_k",
    "reciprocal_rank",
]
