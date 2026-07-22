"""Generation eval with RAGAS — `make eval-rag`.

Runs the full pipeline over a subset of the eval set, scores the answers with the
four RAGAS metrics (Gemini judge + BGE embeddings), prints them, and logs the run
to MLflow. Needs Postgres + a seeded corpus + the ``eval`` extra, and spends a
little Gemini free-tier budget.
"""

from __future__ import annotations

from pathlib import Path

from scholarrag.config import get_settings
from scholarrag.db.engine import session_scope
from scholarrag.eval import load_examples
from scholarrag.eval.mlflow_tracking import log_ragas_run
from scholarrag.eval.ragas_eval import build_judge, collect_samples, run_ragas_eval
from scholarrag.observability import configure_observability
from scholarrag.observability import flush as flush_observability
from scholarrag.pipeline import build_query_engine

EVAL_DIR = Path(__file__).resolve().parents[3] / "data" / "eval"


def main() -> None:  # pragma: no cover - manual entry point
    settings = get_settings()
    configure_observability(settings)  # eval runs produce Langfuse traces too
    examples = load_examples(EVAL_DIR / "golden.json")
    synthetic = EVAL_DIR / "synthetic.json"
    if synthetic.exists():
        examples += load_examples(synthetic)
    examples = examples[: settings.eval_sample_size]  # subset to bound free-tier cost

    engine = build_query_engine(settings)
    with session_scope() as session:
        samples = collect_samples(engine, session, examples)

    judge_llm, judge_embeddings = build_judge(settings)
    scores = run_ragas_eval(
        samples, llm=judge_llm, embeddings=judge_embeddings, max_workers=settings.eval_max_workers
    )

    print(f"RAGAS generation eval over {len(samples)} examples:")
    for name, value in scores.items():
        print(f"  {name:20} {value:.3f}")

    log_ragas_run(
        params={
            "judge_model": settings.gemini_model_cheap,
            "generator_model": settings.gemini_model_strong,
            "reranker": settings.reranker_provider,
            "query_rewriting": settings.query_rewriting_enabled,
            "num_examples": len(samples),
        },
        metrics=scores,
        tracking_uri=settings.mlflow_tracking_uri,
    )
    print("logged run to MLflow")
    flush_observability()  # send buffered traces before the script exits


if __name__ == "__main__":  # pragma: no cover
    main()
