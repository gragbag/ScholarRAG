"""Answer-cache tests — hermetic via an in-memory fake Redis.

Serialization + ``clear`` pass now. The three exercise targets are skipped until
you implement exercises A (exact layer), B (semantic layer), C (engine wiring).
"""

from __future__ import annotations

import fnmatch
import uuid
from collections.abc import Iterator

from sqlalchemy.orm import Session

from scholarrag.cache import AnswerCache, deserialize_answer, serialize_answer
from scholarrag.embeddings import FakeEmbedder
from scholarrag.generation import Answerer
from scholarrag.generation.base import Answer
from scholarrag.llm import FakeLLM
from scholarrag.pipeline import QueryEngine
from scholarrag.retrieval import RetrievedChunk
from scholarrag.retrieval.rewrite import QueryRewriter


class FakeRedis:
    """The minimal Redis surface the cache uses: get/setex/scan_iter/delete."""

    def __init__(self) -> None:
        self.store: dict[str, str] = {}

    def get(self, key: str) -> str | None:
        return self.store.get(key)

    def setex(self, key: str, ttl: int, value: str) -> None:
        self.store[key] = value  # TTL ignored in the fake

    def scan_iter(self, match: str = "*") -> Iterator[str]:
        yield from [k for k in list(self.store) if fnmatch.fnmatch(k, match)]

    def delete(self, *keys: str) -> int:
        n = 0
        for key in keys:
            n += 1 if self.store.pop(key, None) is not None else 0
        return n


def _chunk(text: str) -> RetrievedChunk:
    return RetrievedChunk(
        id="d:0", document_id=uuid.uuid4(), chunk_index=0, text=text, filename="a.txt", score=0.5
    )


def _answer(text: str = "RAG grounds answers [1].") -> Answer:
    return Answer(text=text, sources=[_chunk("RAG grounds answers")])


def _cache(redis: FakeRedis, *, threshold: float = 0.9, fingerprint: str = "cfg") -> AnswerCache:
    return AnswerCache(
        redis,
        FakeEmbedder(dim=64),
        ttl_seconds=60,
        semantic_threshold=threshold,
        config_fingerprint=fingerprint,
    )


def test_answer_serialization_roundtrip() -> None:
    original = _answer()
    restored = deserialize_answer(serialize_answer(original))
    assert restored == original  # frozen dataclasses compare by value


def test_clear_removes_all_entries() -> None:
    redis = FakeRedis()
    redis.store["cache:exact:abc"] = "x"
    redis.store["cache:semantic:def"] = "y"
    redis.store["unrelated"] = "keep"
    assert _cache(redis).clear() == 2
    assert list(redis.store) == ["unrelated"]


# ── Exercise A — the exact layer ─────────────────────────────────────────────
def test_exact_cache_hit_miss_and_config_isolation() -> None:
    redis = FakeRedis()
    cache = _cache(redis, fingerprint="config-1")

    assert cache.get("what is RAG") is None  # cold miss
    cache.put("what is RAG", _answer())
    hit = cache.get("what is RAG")
    assert hit is not None and hit.text == "RAG grounds answers [1]."  # exact hit

    # Same query under a DIFFERENT config must not hit config-1's entry.
    other_config = _cache(redis, fingerprint="config-2")
    assert other_config._get_exact("what is RAG") is None


# ── Exercise B — the semantic layer ──────────────────────────────────────────
def test_semantic_cache_hits_paraphrase_not_unrelated() -> None:
    redis = FakeRedis()
    # FakeEmbedder is bag-of-words: shared words => high cosine similarity.
    cache = _cache(redis, threshold=0.6)
    cache.put("how does retrieval augmented generation reduce hallucination", _answer())

    # Paraphrase sharing most words -> semantic hit (exact key differs).
    hit = cache.get("how does retrieval augmented generation cut hallucination")
    assert hit is not None and hit.text == "RAG grounds answers [1]."

    # Unrelated question -> below threshold -> miss.
    assert cache.get("best shortbread recipe with butter") is None


# ── Exercise C — cache-aside wiring in QueryEngine.query ─────────────────────
def test_query_engine_uses_cache() -> None:
    class StubRetriever:
        def __init__(self) -> None:
            self.calls = 0

        def retrieve(
            self, session: Session, query: str, *, top_k: int = 10
        ) -> list[RetrievedChunk]:
            self.calls += 1
            return [_chunk("RAG grounds answers")]

    retriever = StubRetriever()
    answer_llm = FakeLLM(["RAG grounds answers [1]."])  # ONE scripted answer only
    engine = QueryEngine(
        rewriter=QueryRewriter(llm=FakeLLM(default="")),
        retriever=retriever,
        answerer=Answerer(llm=answer_llm),
        cache=_cache(FakeRedis()),
        top_k=2,
        multi_query=False,
    )

    first = engine.query(Session(), "what is RAG")
    second = engine.query(Session(), "what is RAG")  # must come from the cache

    assert first.text == second.text == "RAG grounds answers [1]."
    assert retriever.calls == 1  # second call skipped retrieval...
    assert len(answer_llm.calls) == 1  # ...and the LLM
