"""The answer cache — exact + semantic layers over Redis.

Skips the whole pipeline (retrieval AND the 6-12s Gemini call) when a question
has effectively been answered before:

* **Exact layer** — key = SHA-256 of ``query + config fingerprint``; identical
  queries return the stored :class:`Answer` in milliseconds.
* **Semantic layer** — the incoming query is embedded (same BGE as retrieval)
  and compared against embeddings of previously answered queries; a cosine
  similarity above the threshold returns that answer, catching *paraphrases*.

Invalidation is TTL-based (entries expire after ``ttl_seconds``), plus
:meth:`clear` which the seed script calls after re-ingesting — a corpus change
makes cached answers stale. (Production upgrade: a corpus-version component in
the key, bumped per ingest.)

Scale note (honest): the semantic lookup scans all cached entries in Python —
O(n) per miss. Fine for this project's scale; at real scale you'd index query
embeddings (Redis VSS, GPTCache, or a vector-store namespace).
"""

from __future__ import annotations

import hashlib
import json
import uuid
from typing import Any

from scholarrag.embeddings.base import Embedder
from scholarrag.generation.base import Answer
from scholarrag.retrieval.base import RetrievedChunk

# Key prefixes: exact entries, semantic entries (kept separate for scan/clear).
_EXACT_PREFIX = "cache:exact:"
_SEMANTIC_PREFIX = "cache:semantic:"


def serialize_answer(answer: Answer) -> str:
    """Answer -> JSON string (UUIDs as text) for storage in Redis."""
    return json.dumps(
        {
            "text": answer.text,
            "sources": [
                {
                    "id": c.id,
                    "document_id": str(c.document_id),
                    "chunk_index": c.chunk_index,
                    "text": c.text,
                    "filename": c.filename,
                    "score": c.score,
                }
                for c in answer.sources
            ],
        }
    )


def deserialize_answer(raw: str | bytes) -> Answer:
    """JSON string -> Answer (inverse of :func:`serialize_answer`)."""
    data = json.loads(raw)
    return Answer(
        text=data["text"],
        sources=[
            RetrievedChunk(
                id=s["id"],
                document_id=uuid.UUID(s["document_id"]),
                chunk_index=s["chunk_index"],
                text=s["text"],
                filename=s["filename"],
                score=s["score"],
            )
            for s in data["sources"]
        ],
    )


class AnswerCache:
    """Two-layer (exact + semantic) answer cache backed by a Redis client.

    ``redis_client`` only needs ``get``/``setex``/``scan_iter``/``delete`` —
    tests inject a tiny in-memory fake instead of a live server.
    """

    def __init__(
        self,
        redis_client: Any,
        embedder: Embedder,
        *,
        ttl_seconds: int = 3600,
        semantic_threshold: float = 0.93,
        semantic_enabled: bool = True,
        config_fingerprint: str = "",
    ) -> None:
        self._redis = redis_client
        self._embedder = embedder
        self._ttl = ttl_seconds
        self._threshold = semantic_threshold
        self._semantic_enabled = semantic_enabled
        # Answers depend on the pipeline config (model, top_k, reranker...) —
        # bake a fingerprint into keys so a config change can't serve stale answers.
        self._fingerprint = config_fingerprint

    # ── exact layer ──────────────────────────────────────────────────────────

    def _exact_key(self, query: str) -> str:
        "Redis key for the exact layer."
        digest = hashlib.sha256(f"{query}|{self._fingerprint}".encode()).hexdigest()
        return _EXACT_PREFIX + digest

    def _get_exact(self, query: str) -> Answer | None:
        "Exact-layer lookup."
        raw = self._redis.get(self._exact_key(query))
        return deserialize_answer(raw) if raw is not None else None

    # ── semantic layer ───────────────────────────────────────────────────────

    def _semantic_lookup(self, query: str) -> Answer | None:
        "Semantic-layer lookup: nearest previously-answered query above threshold."
        query_vec = self._embedder.embed_query(query)
        best_score, best_answer = 0.0, None
        for key in self._redis.scan_iter(match=_SEMANTIC_PREFIX + "*"):
            raw = self._redis.get(key)
            if raw is None:
                continue
            entry = json.loads(raw)
            score = sum(a * b for a, b in zip(query_vec, entry["embedding"], strict=True))
            if score > best_score:
                best_score, best_answer = score, entry["answer"]

        if best_answer is not None and best_score >= self._threshold:
            return deserialize_answer(best_answer)
        return None

    # ── public API (scaffolded) ──────────────────────────────────────────────

    def get(self, query: str) -> Answer | None:
        """Return a cached answer for ``query`` — exact first, then semantic."""
        hit = self._get_exact(query)
        if hit is not None:
            return hit
        if self._semantic_enabled:
            return self._semantic_lookup(query)
        return None

    def put(self, query: str, answer: Answer) -> None:
        """Store ``answer`` in both layers (with TTL)."""
        payload = serialize_answer(answer)
        self._redis.setex(self._exact_key(query), self._ttl, payload)
        if self._semantic_enabled:
            entry = json.dumps({"embedding": self._embedder.embed_query(query), "answer": payload})
            digest = hashlib.sha256(f"{query}|{self._fingerprint}".encode()).hexdigest()
            self._redis.setex(_SEMANTIC_PREFIX + digest, self._ttl, entry)

    def clear(self) -> int:
        """Drop every cached answer (called after re-ingesting the corpus)."""
        keys = [
            key
            for prefix in (_EXACT_PREFIX, _SEMANTIC_PREFIX)
            for key in self._redis.scan_iter(match=prefix + "*")
        ]
        if keys:
            self._redis.delete(*keys)
        return len(keys)
