"""Run a retriever over the eval dataset and aggregate the metrics.

Because it takes any :class:`~scholarrag.retrieval.base.Retriever`, you evaluate
hybrid, dense-only, or rerank-off just by passing a different retriever — that's
how the config comparisons work, all with no LLM cost.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy.orm import Session

from scholarrag.eval.dataset import EvalExample
from scholarrag.eval.retrieval_metrics import (
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from scholarrag.retrieval.base import RetrievedChunk, Retriever


@dataclass(frozen=True, slots=True)
class RetrievalReport:
    """Mean retrieval metrics across the dataset."""

    recall_at_k: float
    precision_at_k: float
    mrr: float
    ndcg_at_k: float
    num_queries: int


def _ranked_filenames(chunks: list[RetrievedChunk]) -> list[str]:
    """Reduce ranked chunks to a ranked list of *distinct* filenames (best rank kept)."""
    seen: set[str] = set()
    ranked: list[str] = []
    for chunk in chunks:
        if chunk.filename not in seen:
            seen.add(chunk.filename)
            ranked.append(chunk.filename)
    return ranked


def evaluate_retrieval(
    retriever: Retriever,
    session: Session,
    examples: list[EvalExample],
    *,
    k: int = 5,
) -> RetrievalReport:
    """Score ``retriever`` over ``examples`` and return the averaged report."""
    recalls: list[float] = []
    precisions: list[float] = []
    rrs: list[float] = []
    ndcgs: list[float] = []

    for example in examples:
        chunks = retriever.retrieve(session, example.question, top_k=k)
        ranked = _ranked_filenames(chunks)
        relevant = set(example.relevant_files)
        recalls.append(recall_at_k(ranked, relevant, k))
        precisions.append(precision_at_k(ranked, relevant, k))
        rrs.append(reciprocal_rank(ranked, relevant))
        ndcgs.append(ndcg_at_k(ranked, relevant, k))

    n = len(examples)
    mean = (lambda xs: sum(xs) / n) if n else (lambda xs: 0.0)
    return RetrievalReport(
        recall_at_k=mean(recalls),
        precision_at_k=mean(precisions),
        mrr=mean(rrs),
        ndcg_at_k=mean(ndcgs),
        num_queries=n,
    )
