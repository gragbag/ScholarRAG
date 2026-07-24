"""Agentic vs single-shot on the hard set — `make eval-agentic`.

Runs BOTH pipelines (langchain, agentic) over data/eval/hard.json, prints the
comparison table, and logs one MLflow row per pipeline. Needs a seeded corpus,
the langchain+agentic extras, and a Gemini key; several rate-limited minutes.

Cache is force-disabled for the run: the answer-cache fingerprint would
otherwise let the second pipeline serve the first one's cached answers.
"""

from __future__ import annotations

from pathlib import Path

from scholarrag.config import get_settings
from scholarrag.db.engine import session_scope
from scholarrag.eval.compare import (
    build_engine_with_counter,
    collect_results,
    load_hard_examples,
    score_results,
)
from scholarrag.eval.mlflow_tracking import log_ragas_run

HARD_SET = Path(__file__).resolve().parents[3] / "data" / "eval" / "hard.json"


def main() -> None:  # pragma: no cover - manual entry point
    base = get_settings()
    examples = load_hard_examples(HARD_SET)
    print(
        f"hard set: {len(examples)} questions "
        f"({sum(e.answerable for e in examples)} answerable, "
        f"{sum(not e.answerable for e in examples)} controls)\n"
    )

    for kind in ("langchain", "agentic"):
        # Generation runs on the CHEAP tier here, not the usual strong model.
        # The strong model (gemini-3.5-flash) is free-tier-capped at 20 requests/DAY;
        # a full both-pipelines run needs ~22 generate calls and can't fit. flash-lite
        # has a far larger daily quota. Both pipelines use the identical model, so the
        # agentic-vs-single-shot comparison stays fair. (Documented in BENCHMARKS.)
        settings = base.model_copy(
            update={
                "pipeline": kind,
                "cache_enabled": False,
                "gemini_model_strong": base.gemini_model_cheap,
            }
        )
        engine, counter = build_engine_with_counter(settings)
        with session_scope() as session:
            results = collect_results(engine, session, examples, counter)
        report = score_results(results)

        print(f"── {kind} ──")
        print(f"  grounded-answer rate (answerable): {report.answered_rate:.2f}")
        print(f"  false-answer rate (controls):      {report.false_answer_rate:.2f}")
        print(f"  source-hit rate:                   {report.source_hit_rate:.2f}")
        print(f"  mean LLM calls / query:            {report.mean_llm_calls:.1f}")
        print(f"  mean latency:                      {report.mean_latency_s:.1f}s\n")

        log_ragas_run(
            params={"pipeline": kind, "dataset": "hard", "n_questions": len(examples)},
            metrics={
                "answered_rate": report.answered_rate,
                "false_answer_rate": report.false_answer_rate,
                "source_hit_rate": report.source_hit_rate,
                "mean_llm_calls": report.mean_llm_calls,
                "mean_latency_s": report.mean_latency_s,
            },
            tracking_uri=base.mlflow_tracking_uri,
            experiment="scholarrag-agentic-eval",
        )
    print("logged both runs to MLflow (experiment: scholarrag-agentic-eval)")


if __name__ == "__main__":  # pragma: no cover
    main()
