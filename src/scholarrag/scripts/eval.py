"""Run retrieval eval over the golden (+ synthetic) set — `make eval`.

Scores the hybrid retriever against the labelled dataset and prints the metrics.
Needs Postgres (lexical) + the real embedder/vector store, so it runs the actual
configured retrieval stack — the numbers here are what belong in BENCHMARKS.md.
"""

from __future__ import annotations

from pathlib import Path

from scholarrag.config import get_settings
from scholarrag.db.engine import session_scope
from scholarrag.eval import evaluate_retrieval, load_examples
from scholarrag.retrieval import build_hybrid_retriever

EVAL_DIR = Path(__file__).resolve().parents[3] / "data" / "eval"


def main() -> None:  # pragma: no cover - manual entry point
    settings = get_settings()
    examples = load_examples(EVAL_DIR / "golden.json")
    synthetic = EVAL_DIR / "synthetic.json"
    if synthetic.exists():
        examples += load_examples(synthetic)

    retriever = build_hybrid_retriever(settings)
    with session_scope() as session:
        report = evaluate_retrieval(retriever, session, examples, k=settings.eval_k)

    k = settings.eval_k
    print(f"queries       : {report.num_queries}")
    print(f"Recall@{k}     : {report.recall_at_k:.3f}")
    print(f"Precision@{k}  : {report.precision_at_k:.3f}")
    print(f"MRR           : {report.mrr:.3f}")
    print(f"nDCG@{k}       : {report.ndcg_at_k:.3f}")
    if report.recall_at_k < settings.eval_recall_threshold:
        print(f"WARNING: Recall@{k} below threshold {settings.eval_recall_threshold}")


if __name__ == "__main__":  # pragma: no cover
    main()
