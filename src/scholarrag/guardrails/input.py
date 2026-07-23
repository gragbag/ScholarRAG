"""Input guardrails — query hygiene and per-client rate limiting.

Length bounds live on the request model itself (Pydantic ``Field`` in the query
route -> automatic 422s). Here: sanitization for what gets past the schema, and
a Redis fixed-window rate limiter so an open endpoint can't be looped into a
token-budget attack (every uncached query is ~2.5K tokens + a 5-RPM budget).
"""

from __future__ import annotations

import re
import time
from typing import Any

# Strip C0 control chars (keep tab/newline) — same hygiene as PDF extraction.
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def sanitize_query(query: str) -> str:
    """Normalize an incoming query: drop control bytes, collapse outer whitespace."""
    return _CONTROL_CHARS_RE.sub("", query).strip()


class RateLimiter:
    """Fixed-window request limiter on Redis: N requests per client per minute.

    ``redis_client`` needs only ``incr`` and ``expire`` — tests inject a fake.
    """

    def __init__(self, redis_client: Any, *, per_minute: int = 20) -> None:
        self._redis = redis_client
        self._limit = per_minute

    def allow(self, client_id: str) -> bool:
        "True if this request fits the client's budget for the current window."

        window = int(time.time() // 60)
        key = f"ratelimit:{client_id}:{window}"

        count: int = self._redis.incr(key)

        if count == 1:
            self._redis.expire(key, 120)

        return count <= self._limit
