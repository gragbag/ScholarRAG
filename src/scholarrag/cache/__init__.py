"""Cache package — exact + semantic answer caching on Redis (Phase 4 Step 2).

A cache hit skips the whole pipeline (retrieval AND the LLM call). Use
:func:`build_answer_cache` to get the cache implied by settings — ``None`` when
disabled, so the pipeline works identically without it.
"""

from __future__ import annotations

from scholarrag.cache.answer_cache import AnswerCache, deserialize_answer, serialize_answer
from scholarrag.config import Settings
from scholarrag.embeddings.base import Embedder

__all__ = [
    "AnswerCache",
    "build_answer_cache",
    "deserialize_answer",
    "serialize_answer",
]


def _config_fingerprint(settings: Settings) -> str:
    """The pipeline knobs an answer depends on — part of every cache key."""
    return "|".join(
        str(v)
        for v in (
            settings.llm_provider,
            settings.llm_model_strong,
            settings.gemini_model_strong,
            settings.retrieval_top_k,
            settings.reranker_provider,
            settings.query_rewriting_enabled,
            settings.corpus_profile,
        )
    )


def build_answer_cache(settings: Settings, embedder: Embedder) -> AnswerCache | None:
    """Return the configured :class:`AnswerCache`, or ``None`` when disabled."""
    if not settings.cache_enabled:
        return None
    import redis  # lazy: only needed when the cache is actually on

    client = redis.Redis.from_url(settings.redis_url)
    return AnswerCache(
        client,
        embedder,
        ttl_seconds=settings.cache_ttl_seconds,
        semantic_threshold=settings.semantic_cache_threshold,
        semantic_enabled=settings.semantic_cache_enabled,
        config_fingerprint=_config_fingerprint(settings),
    )
