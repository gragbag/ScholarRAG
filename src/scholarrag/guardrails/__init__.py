"""Guardrails package — policy at the boundaries (Phase 4 Step 3).

Input side: query sanitization + a Redis fixed-window rate limiter. Output side:
the grounding gate (no citations and no honest refusal -> refuse, don't guess).
Prompt-injection hardening lives in ``generation/prompts.py`` (delimited sources
+ the untrusted-data rule).
"""

from __future__ import annotations

from scholarrag.config import Settings
from scholarrag.guardrails.input import RateLimiter, sanitize_query
from scholarrag.guardrails.output import REFUSAL_MESSAGE, enforce_grounding, looks_like_refusal

__all__ = [
    "REFUSAL_MESSAGE",
    "RateLimiter",
    "build_rate_limiter",
    "enforce_grounding",
    "looks_like_refusal",
    "sanitize_query",
]


def build_rate_limiter(settings: Settings) -> RateLimiter | None:
    """Return the configured rate limiter, or ``None`` when disabled."""
    if not settings.rate_limit_enabled:
        return None
    import redis  # lazy: only needed when limiting is actually on

    client = redis.Redis.from_url(settings.redis_url)
    return RateLimiter(client, per_minute=settings.rate_limit_per_minute)
