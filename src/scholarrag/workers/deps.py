"""Per-worker dependencies.

The embedder loads a real model (BGE) on first use — expensive. A worker should
build the pipeline **once per process** and reuse it across every task, not
rebuild it per task. ``lru_cache`` gives us exactly that: one instance per
process, created lazily.
"""

from __future__ import annotations

from functools import lru_cache

from scholarrag.config import get_settings
from scholarrag.embeddings import build_embedder
from scholarrag.ingestion import IngestionPipeline
from scholarrag.vectorstore import build_vector_store


@lru_cache
def get_pipeline() -> IngestionPipeline:
    """Return the process-wide ingestion pipeline (built from settings once)."""
    settings = get_settings()
    return IngestionPipeline(
        embedder=build_embedder(settings),
        vector_store=build_vector_store(settings),
    )
