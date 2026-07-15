"""Embedder tests.

The interface/factory tests below pass now. The behaviour tests are skipped
until you implement ``FakeEmbedder._embed`` (Step 2 exercise) — remove each
``@pytest.mark.skip`` to turn them on.

None of these load a real model: the factory constructs ``LocalEmbedder`` lazily
(no torch import), and everything else uses the dependency-free ``FakeEmbedder``.
"""

from __future__ import annotations

import math

import pytest

from scholarrag.config import Settings
from scholarrag.embeddings import (
    Embedder,
    FakeEmbedder,
    LocalEmbedder,
    Vector,
    build_embedder,
)


def _cosine(a: Vector, b: Vector) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return 0.0 if na == 0 or nb == 0 else dot / (na * nb)


# ── Interface + factory (pass now) ─────────────────────────────────────────
def test_fake_embedder_conforms_to_protocol() -> None:
    emb = FakeEmbedder(dim=8)
    assert isinstance(emb, Embedder)
    assert emb.dim == 8


def test_build_embedder_returns_fake() -> None:
    emb = build_embedder(Settings(embedding_provider="fake", embedding_dim=16))
    assert isinstance(emb, FakeEmbedder)
    assert emb.dim == 16


def test_build_embedder_returns_local_without_loading_model() -> None:
    # Constructing LocalEmbedder must be lazy — no torch import, no download.
    emb = build_embedder(
        Settings(
            embedding_provider="local",
            embedding_model="BAAI/bge-small-en-v1.5",
            embedding_dim=384,
        )
    )
    assert isinstance(emb, LocalEmbedder)
    assert emb.dim == 384


def test_build_embedder_unknown_provider_raises() -> None:
    with pytest.raises(ValueError, match="unsupported embedding provider"):
        build_embedder(Settings(embedding_provider="openai"))


# ── Behaviour (Step 2 exercise) ─────────
def test_fake_embed_shape_and_normalized() -> None:
    emb = FakeEmbedder(dim=32)
    v = emb.embed_query("neural networks and attention")
    assert len(v) == 32
    assert math.isclose(math.sqrt(sum(x * x for x in v)), 1.0, rel_tol=1e-6)


def test_fake_embed_is_deterministic() -> None:
    emb = FakeEmbedder(dim=32)
    assert emb.embed_query("hello world") == emb.embed_query("hello world")


def test_fake_embed_shared_words_more_similar() -> None:
    emb = FakeEmbedder(dim=64)
    a = emb.embed_query("transformers use attention mechanisms")
    b = emb.embed_query("attention mechanisms in transformers")  # same words, reordered
    c = emb.embed_query("the price of tea in china")  # disjoint words
    assert _cosine(a, b) > _cosine(a, c)


def test_fake_embed_documents_batch() -> None:
    emb = FakeEmbedder(dim=16)
    vectors = emb.embed_documents(["one two", "three four five"])
    assert len(vectors) == 2
    assert all(len(v) == 16 for v in vectors)


def test_fake_embed_empty_text_is_zero_vector() -> None:
    emb = FakeEmbedder(dim=8)
    assert emb.embed_query("") == [0.0] * 8
