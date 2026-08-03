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
    ModalEmbedder,
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


# ── ModalEmbedder (BGE hosted on Modal — deploy path, no torch) ───────────────
# The post_fn seam captures the request, so these need no Modal, URL, or network.
def _modal_settings() -> Settings:
    return Settings(
        _env_file=None,
        modal_embed_url="https://modal.example/embed",
        modal_embed_token="tok",
        embedding_dim=2,
    )


def test_modal_embedder_conforms_and_reports_dim() -> None:
    emb = ModalEmbedder(_modal_settings(), post_fn=lambda *a, **k: {"embeddings": []})
    assert isinstance(emb, Embedder)
    assert emb.dim == 2


def test_modal_embedder_embeds_documents_with_auth() -> None:
    seen: dict[str, object] = {}

    def fake_post(
        url: str, *, json: dict[str, object], headers: dict[str, str]
    ) -> dict[str, object]:
        seen["url"] = url
        seen["json"] = json
        seen["headers"] = headers
        return {"embeddings": [[0.1, 0.2]] * len(json["texts"])}  # type: ignore[arg-type]

    emb = ModalEmbedder(_modal_settings(), query_prefix="Q: ", post_fn=fake_post)

    assert emb.embed_documents(["a", "b"]) == [[0.1, 0.2], [0.1, 0.2]]
    assert seen["url"] == "https://modal.example/embed"
    assert seen["json"] == {"texts": ["a", "b"]}  # documents sent verbatim
    assert seen["headers"]["Authorization"] == "Bearer tok"  # type: ignore[index]


def test_modal_embedder_query_gets_prefix_and_single_vector() -> None:
    seen: dict[str, object] = {}

    def fake_post(
        url: str, *, json: dict[str, object], headers: dict[str, str]
    ) -> dict[str, object]:
        seen["json"] = json
        return {"embeddings": [[1.0, 2.0]]}

    emb = ModalEmbedder(_modal_settings(), query_prefix="Q: ", post_fn=fake_post)

    assert emb.embed_query("hello") == [1.0, 2.0]  # unwrapped from the batch
    assert seen["json"] == {"texts": ["Q: hello"]}  # BGE query prefix applied


def test_modal_embedder_empty_documents_short_circuits() -> None:
    def boom(*a: object, **k: object) -> dict[str, object]:
        raise AssertionError("should not call the endpoint for an empty batch")

    emb = ModalEmbedder(_modal_settings(), post_fn=boom)
    assert emb.embed_documents([]) == []
