"""Log eval runs to MLflow so configs can be compared over time.

Each run records the *config* (params: judge model, reranker, sample size) and
the *scores* (metrics: the RAGAS numbers), so the MLflow UI (docker-compose, port
5001) becomes an experiment log — "rerank on vs off" as two comparable rows.
"""

from __future__ import annotations

from typing import Any


def log_ragas_run(
    *,
    params: dict[str, Any],
    metrics: dict[str, float],
    tracking_uri: str | None = None,
    experiment: str = "scholarrag-generation-eval",
) -> None:
    """Record one generation-eval run (params + metrics) to MLflow."""
    import mlflow  # lazy: only needed when actually logging (eval extra)

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)
    with mlflow.start_run():
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
