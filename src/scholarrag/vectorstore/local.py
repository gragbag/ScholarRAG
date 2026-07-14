"""In-process vector store used for tests, CI, and local dev.

Cosine-similarity brute-force search over NumPy arrays. Correct and fully
deterministic, which is exactly what we want for a hermetic test suite. It is
not built for scale — that is Pinecone's job in production.
"""

from __future__ import annotations

import numpy as np

from scholarrag.vectorstore.base import (
    Metadata,
    QueryMatch,
    VectorRecord,
)


def _matches_filter(metadata: Metadata, flt: Metadata | None) -> bool:
    if not flt:
        return True
    return all(metadata.get(key) == value for key, value in flt.items())


class LocalVectorStore:
    """A tiny cosine-similarity store, namespaced by string key."""

    def __init__(self, *, dim: int) -> None:
        self._dim = dim
        # namespace -> id -> (vector, metadata)
        self._data: dict[str, dict[str, tuple[np.ndarray, Metadata]]] = {}

    def _ns(self, namespace: str) -> dict[str, tuple[np.ndarray, Metadata]]:
        return self._data.setdefault(namespace, {})

    def upsert(self, records: list[VectorRecord], *, namespace: str = "") -> int:
        ns = self._ns(namespace)
        for record in records:
            if len(record.values) != self._dim:
                raise ValueError(
                    f"vector for id {record.id!r} has dim {len(record.values)}, "
                    f"expected {self._dim}"
                )
            ns[record.id] = (np.asarray(record.values, dtype=np.float32), dict(record.metadata))
        return len(records)

    def query(
        self,
        vector: list[float],
        *,
        top_k: int = 10,
        namespace: str = "",
        filter: Metadata | None = None,
    ) -> list[QueryMatch]:
        ns = self._ns(namespace)
        if not ns:
            return []

        q = np.asarray(vector, dtype=np.float32)
        q_norm = float(np.linalg.norm(q))
        if q_norm == 0.0:
            return []

        scored: list[QueryMatch] = []
        for rec_id, (vec, metadata) in ns.items():
            if not _matches_filter(metadata, filter):
                continue
            denom = q_norm * float(np.linalg.norm(vec))
            score = 0.0 if denom == 0.0 else float(np.dot(q, vec) / denom)
            scored.append(QueryMatch(id=rec_id, score=score, metadata=dict(metadata)))

        scored.sort(key=lambda m: m.score, reverse=True)
        return scored[:top_k]

    def fetch(self, id: str, *, namespace: str = "") -> Metadata | None:
        entry = self._ns(namespace).get(id)
        if entry is None:
            return None

        _vector, metadata = entry
        return dict(metadata)

    def delete(
        self,
        ids: list[str] | None = None,
        *,
        namespace: str = "",
        delete_all: bool = False,
    ) -> int:
        ns = self._ns(namespace)
        if delete_all:
            removed = len(ns)
            ns.clear()
            return removed
        if not ids:
            return 0
        removed = 0
        for rec_id in ids:
            if ns.pop(rec_id, None) is not None:
                removed += 1
        return removed

    def count(self, *, namespace: str = "") -> int:
        return len(self._ns(namespace))
