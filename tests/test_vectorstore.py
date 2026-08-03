"""Tests for the vector-store abstraction and its local implementation."""

from __future__ import annotations

import math

import pytest
from sqlalchemy.engine import Engine

from scholarrag.config import Settings
from scholarrag.vectorstore import (
    LocalVectorStore,
    VectorRecord,
    VectorStore,
    build_vector_store,
)
from scholarrag.vectorstore.pgvector import PgVectorStore


@pytest.fixture
def store() -> LocalVectorStore:
    return LocalVectorStore(dim=3)


def test_local_store_satisfies_protocol(store: LocalVectorStore) -> None:
    assert isinstance(store, VectorStore)


def test_upsert_and_count(store: LocalVectorStore) -> None:
    written = store.upsert(
        [
            VectorRecord(id="a", values=[1.0, 0.0, 0.0], metadata={"doc": "1"}),
            VectorRecord(id="b", values=[0.0, 1.0, 0.0], metadata={"doc": "2"}),
        ]
    )
    assert written == 2
    assert store.count() == 2


def test_query_ranks_by_cosine_similarity(store: LocalVectorStore) -> None:
    store.upsert(
        [
            VectorRecord(id="close", values=[1.0, 0.1, 0.0]),
            VectorRecord(id="far", values=[0.0, 0.0, 1.0]),
        ]
    )
    matches = store.query([1.0, 0.0, 0.0], top_k=2)
    assert [m.id for m in matches] == ["close", "far"]
    assert matches[0].score > matches[1].score
    assert math.isclose(matches[0].score, 1.0 / math.sqrt(1.01), rel_tol=1e-6)


def test_query_top_k_limits_results(store: LocalVectorStore) -> None:
    store.upsert([VectorRecord(id=str(i), values=[float(i), 1.0, 0.0]) for i in range(5)])
    assert len(store.query([1.0, 1.0, 0.0], top_k=2)) == 2


def test_query_metadata_filter(store: LocalVectorStore) -> None:
    store.upsert(
        [
            VectorRecord(id="a", values=[1.0, 0.0, 0.0], metadata={"lang": "en"}),
            VectorRecord(id="b", values=[1.0, 0.0, 0.0], metadata={"lang": "fr"}),
        ]
    )
    matches = store.query([1.0, 0.0, 0.0], top_k=5, filter={"lang": "fr"})
    assert [m.id for m in matches] == ["b"]


def test_query_empty_store_returns_empty(store: LocalVectorStore) -> None:
    assert store.query([1.0, 0.0, 0.0]) == []


def test_namespaces_are_isolated(store: LocalVectorStore) -> None:
    store.upsert([VectorRecord(id="a", values=[1.0, 0.0, 0.0])], namespace="one")
    store.upsert([VectorRecord(id="b", values=[0.0, 1.0, 0.0])], namespace="two")
    assert store.count(namespace="one") == 1
    assert store.count(namespace="two") == 1
    assert [m.id for m in store.query([1.0, 0.0, 0.0], namespace="two")] == ["b"]


def test_upsert_dimension_mismatch_raises(store: LocalVectorStore) -> None:
    with pytest.raises(ValueError, match="expected 3"):
        store.upsert([VectorRecord(id="bad", values=[1.0, 0.0])])


def test_fetch_returns_metadata(store: LocalVectorStore) -> None:
    store.upsert([VectorRecord(id="a", values=[1.0, 0.0, 0.0], metadata={"doc": "1"})])
    assert store.fetch("a") == {"doc": "1"}


def test_fetch_missing_returns_none(store: LocalVectorStore) -> None:
    assert store.fetch("nope") is None


def test_delete_by_id(store: LocalVectorStore) -> None:
    store.upsert(
        [
            VectorRecord(id="a", values=[1.0, 0.0, 0.0]),
            VectorRecord(id="b", values=[0.0, 1.0, 0.0]),
        ]
    )
    assert store.delete(["a", "missing"]) == 1
    assert store.count() == 1


def test_delete_all(store: LocalVectorStore) -> None:
    store.upsert([VectorRecord(id="a", values=[1.0, 0.0, 0.0])])
    assert store.delete(delete_all=True) == 1
    assert store.count() == 0


def test_build_vector_store_defaults_to_local() -> None:
    settings = Settings(vector_store="auto", pinecone_api_key=None, embedding_dim=3)
    assert isinstance(build_vector_store(settings), LocalVectorStore)


# ── PgVectorStore (consolidated pgvector backend) ────────────────────────────
# These need a real Postgres with pgvector. The shared db_engine fixture creates
# the extension + embeddings table and skips if Postgres is unreachable. Writes
# commit (not savepoint-isolated), so the fixture wipes the table first.
_DIM = 384


def _onehot(index: int) -> list[float]:
    """A 384-dim one-hot vector (matches the embeddings.embedding column dim)."""
    v = [0.0] * _DIM
    v[index] = 1.0
    return v


@pytest.fixture
def pg_store(db_engine: Engine) -> PgVectorStore:
    store = PgVectorStore(db_engine, dim=_DIM)
    store.delete(delete_all=True)  # clean slate
    return store


def test_pgvector_satisfies_protocol(pg_store: PgVectorStore) -> None:
    assert isinstance(pg_store, VectorStore)


def test_pgvector_upsert_count_fetch_delete(pg_store: PgVectorStore) -> None:
    written = pg_store.upsert(
        [
            VectorRecord(id="a", values=_onehot(0), metadata={"collection": "x"}),
            VectorRecord(id="b", values=_onehot(1), metadata={"collection": "y"}),
        ]
    )
    assert written == 2
    assert pg_store.count() == 2
    assert pg_store.fetch("a") == {"collection": "x"}
    assert pg_store.fetch("missing") is None

    assert pg_store.delete(["a"]) == 1
    assert pg_store.count() == 1


def test_pgvector_query_knn_and_filter(pg_store: PgVectorStore) -> None:
    pg_store.upsert(
        [
            VectorRecord(id="a", values=_onehot(0), metadata={"collection": "x"}),
            VectorRecord(id="b", values=_onehot(1), metadata={"collection": "y"}),
        ]
    )
    # Nearest to the one-hot-0 vector is "a" (cosine sim 1.0 vs 0.0).
    matches = pg_store.query(_onehot(0), top_k=2)
    assert [m.id for m in matches] == ["a", "b"]
    assert matches[0].score > matches[1].score
    assert matches[0].metadata == {"collection": "x"}

    # Metadata containment filter restricts to a folder.
    filtered = pg_store.query(_onehot(0), top_k=5, filter={"collection": "y"})
    assert [m.id for m in filtered] == ["b"]
