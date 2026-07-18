"""Pinecone-backed vector store (production / default).

Only imported when Pinecone is actually selected — see
``scholarrag.vectorstore.build_vector_store`` — so neither tests nor CI need the
``pinecone`` package or a live index.

The connection is established lazily on first use. Index provisioning and
graceful handling of a *paused* free-tier index (Pinecone Starter indexes pause
after ~3 weeks of inactivity) are fleshed out in Phase 1 / Phase 7; the shape is
in place here so the rest of the pipeline can depend on the interface today.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from scholarrag.config import Settings
from scholarrag.vectorstore.base import (
    Metadata,
    QueryMatch,
    VectorRecord,
)

if TYPE_CHECKING:  # pragma: no cover
    from pinecone import Index


class PineconeVectorStore:
    """Serverless Pinecone implementation of the :class:`VectorStore` protocol."""

    def __init__(self, settings: Settings) -> None:
        if settings.pinecone_api_key is None:
            raise ValueError("PINECONE_API_KEY is required for PineconeVectorStore")
        self._settings = settings
        self._index: Index | None = None

    def _connect(self) -> Index:
        """Lazily create the client and (if needed) the index, then return it."""
        if self._index is not None:
            return self._index

        from pinecone import Pinecone, ServerlessSpec

        pc = Pinecone(api_key=self._settings.pinecone_api_key)
        existing = {idx["name"] for idx in pc.list_indexes()}
        if self._settings.pinecone_index not in existing:
            pc.create_index(
                name=self._settings.pinecone_index,
                dimension=self._settings.embedding_dim,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud=self._settings.pinecone_cloud,
                    region=self._settings.pinecone_region,
                ),
            )
        self._index = pc.Index(self._settings.pinecone_index)
        return self._index

    def upsert(self, records: list[VectorRecord], *, namespace: str = "") -> int:
        index = self._connect()
        vectors: list[dict[str, Any]] = [
            {"id": r.id, "values": r.values, "metadata": dict(r.metadata)} for r in records
        ]
        index.upsert(vectors=vectors, namespace=namespace)
        return len(records)

    def query(
        self,
        vector: list[float],
        *,
        top_k: int = 10,
        namespace: str = "",
        filter: Metadata | None = None,
    ) -> list[QueryMatch]:
        index = self._connect()
        result = index.query(
            vector=vector,
            top_k=top_k,
            namespace=namespace,
            filter=filter,
            include_metadata=True,
        )
        return [
            QueryMatch(
                id=str(match["id"]),
                score=float(match["score"]),
                metadata=dict(match.get("metadata") or {}),
            )
            for match in result.get("matches", [])
        ]

    def fetch(self, id: str, *, namespace: str = "") -> Metadata | None:
        index = self._connect()
        result = index.fetch(ids=[id], namespace=namespace)
        vectors = result.get("vectors") or {}
        record = vectors.get(id)
        if record is None:
            return None

        return dict(record.get("metadata") or {})

    def delete(
        self,
        ids: list[str] | None = None,
        *,
        namespace: str = "",
        delete_all: bool = False,
    ) -> int:
        index = self._connect()
        if delete_all:
            index.delete(delete_all=True, namespace=namespace)
            return -1  # Pinecone does not report a count for bulk deletes.
        if not ids:
            return 0
        index.delete(ids=ids, namespace=namespace)
        return len(ids)

    def count(self, *, namespace: str = "") -> int:
        index = self._connect()
        stats = index.describe_index_stats()
        namespaces = stats.get("namespaces") or {}
        # SDK 9.x labels the default namespace "__default__" in the stats, even
        # though we upsert/query it as "". And each entry is a NamespaceSummary
        # object (attribute access), not a dict.
        ns_stats = namespaces.get(namespace or "__default__")
        if ns_stats is None:
            return 0
        vector_count = getattr(ns_stats, "vector_count", None)
        if vector_count is None and isinstance(ns_stats, dict):
            vector_count = ns_stats.get("vector_count", 0)
        return int(vector_count or 0)
