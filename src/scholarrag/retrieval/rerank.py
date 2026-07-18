"""Reranking — the precision second stage of retrieve-and-rerank.

RRF fusion gives a good *candidate set* cheaply, but it never actually re-reads
the text against the query. A **cross-encoder** does: it feeds ``query [SEP]
chunk`` through a transformer *jointly*, so every query token attends to every
chunk token, and scores true relevance. That's far more accurate than the
bi-encoder (embedding) similarity used in stage 1 — but it costs one model
forward pass *per candidate at query time*, so we only run it over the fused
shortlist (~50), then keep the top handful.

Three pieces, mirroring the embeddings package:

* :class:`Reranker`             — the protocol downstream code depends on.
* :class:`CrossEncoderReranker` — real; lazy-loads a sentence-transformers
  ``CrossEncoder`` (the ``embeddings`` extra). Only touched when selected.
* :class:`FakeReranker`         — deterministic, torch-free; used in tests/CI.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import replace
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from scholarrag.retrieval.base import RetrievedChunk

if TYPE_CHECKING:  # pragma: no cover
    from sentence_transformers import CrossEncoder

# A scoring function: (query, passage) pairs -> one relevance score each. The
# real model provides this; tests inject their own to check the reorder logic.
PredictFn = Callable[[list[tuple[str, str]]], list[float]]

_TOKEN_RE = re.compile(r"[a-z0-9]+")


@runtime_checkable
class Reranker(Protocol):
    """Reorders a candidate list by true query relevance, keeping the top ``top_k``."""

    def rerank(
        self, query: str, chunks: list[RetrievedChunk], *, top_k: int = 10
    ) -> list[RetrievedChunk]:
        """Return the ``top_k`` most relevant chunks, best first."""
        ...


class FakeReranker:
    """Deterministic, dependency-free reranker for tests.

    Scores each chunk by how many query word-tokens it contains (a crude
    overlap), so the ordering is predictable and needs no model. Enough to
    exercise the hybrid wiring without loading a cross-encoder.
    """

    def rerank(
        self, query: str, chunks: list[RetrievedChunk], *, top_k: int = 10
    ) -> list[RetrievedChunk]:
        query_terms = set(_TOKEN_RE.findall(query.lower()))
        scored = [
            replace(
                chunk, score=float(len(query_terms & set(_TOKEN_RE.findall(chunk.text.lower()))))
            )
            for chunk in chunks
        ]
        # Stable sort: ties keep their incoming (fused) order.
        scored.sort(key=lambda chunk: chunk.score, reverse=True)
        return scored[:top_k]


class CrossEncoderReranker:
    """Cross-encoder reranker (production). Lazy-loads the model on first use.

    ``predict_fn`` is a test seam: leave it ``None`` in real use (the model is
    loaded on demand); tests pass a stub so the reorder logic can be checked
    without torch.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        *,
        predict_fn: PredictFn | None = None,
    ) -> None:
        self._model_name = model_name
        self._predict_fn = predict_fn
        self._model: CrossEncoder | None = None

    def _load(self) -> CrossEncoder:
        """Load (and cache) the cross-encoder on first use."""
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self._model_name)
        return self._model

    def _predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Score (query, passage) pairs — via the injected fn or the real model."""
        if self._predict_fn is not None:
            return self._predict_fn(pairs)
        model = self._load()
        return [float(score) for score in model.predict(pairs)]

    def rerank(
        self, query: str, chunks: list[RetrievedChunk], *, top_k: int = 10
    ) -> list[RetrievedChunk]:
        "Reorder ``chunks`` by cross-encoder relevance, keeping the top ``top_k``."
        if not chunks:
            return []

        pairs = [(query, chunk.text) for chunk in chunks]
        scores = self._predict(pairs)
        reranked = [replace(chunk, score=float(s)) for chunk, s in zip(chunks, scores, strict=True)]

        reranked.sort(key=lambda chunk: chunk.score, reverse=True)
        return reranked[:top_k]
