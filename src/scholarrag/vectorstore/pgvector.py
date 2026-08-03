"""Postgres/pgvector-backed vector store — the consolidated store for deploy.

Vectors live in the ``embeddings`` table (see the ``Embedding`` model + its
migration), right next to the documents/BM25 data — so one database serves dense
AND lexical retrieval and there's no Pinecone to run. Remember pgvector only
STORES and SEARCHES vectors: the 384-dim embeddings are produced in Python by the
``Embedder`` (BGE now, Modal later), for both documents (ingest) and the query.

Construction does no I/O; each method opens a short transaction on the engine.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from sqlalchemy import bindparam, text

from scholarrag.vectorstore.base import Metadata, QueryMatch, VectorRecord

if TYPE_CHECKING:  # pragma: no cover
    from sqlalchemy.engine import Engine


def _vec_literal(values: list[float]) -> str:
    """Render a vector as a pgvector text literal, e.g. ``[0.1,0.2,0.3]``."""
    return "[" + ",".join(repr(float(v)) for v in values) + "]"


class PgVectorStore:
    """pgvector implementation of the :class:`VectorStore` protocol."""

    def __init__(self, engine: Engine, *, dim: int) -> None:
        self._engine = engine
        self._dim = dim

    def upsert(self, records: list[VectorRecord], *, namespace: str = "") -> int:
        if not records:
            return 0
        stmt = text(
            """
            INSERT INTO embeddings (vector_id, namespace, embedding, metadata)
            VALUES (:id, :ns, CAST(:vec AS vector), CAST(:meta AS jsonb))
            ON CONFLICT (vector_id) DO UPDATE
            SET namespace = EXCLUDED.namespace,
                embedding = EXCLUDED.embedding,
                metadata  = EXCLUDED.metadata
            """
        )
        with self._engine.begin() as conn:
            for r in records:
                if len(r.values) != self._dim:
                    raise ValueError(f"expected dim {self._dim}, got {len(r.values)}")
                conn.execute(
                    stmt,
                    {
                        "id": r.id,
                        "ns": namespace,
                        "vec": _vec_literal(r.values),
                        "meta": json.dumps(dict(r.metadata)),
                    },
                )
        return len(records)

    def query(
        self,
        vector: list[float],
        *,
        top_k: int = 10,
        namespace: str = "",
        filter: Metadata | None = None,
    ) -> list[QueryMatch]:
        "Return up to ``top_k`` nearest neighbours to ``vector`` (cosine)."
        where, params = self._where(namespace, filter)
        params["qvec"] = _vec_literal(vector)
        params["k"] = top_k

        stmt = text(
            f"""
            SELECT vector_id, metadata, 1 - (embedding <=> CAST(:qvec AS vector)) AS score
            FROM embeddings
            WHERE {where}
            ORDER BY embedding <=> CAST(:qvec AS vector)
            LIMIT :k
            """
        )

        with self._engine.connect() as conn:
            return [
                QueryMatch(id=row.vector_id, score=float(row.score), metadata=dict(row.metadata))
                for row in conn.execute(stmt, params)
            ]

    def _where(self, namespace: str, filter: Metadata | None) -> tuple[str, dict[str, Any]]:
        """WHERE fragment + bind params: namespace match, plus JSONB containment."""
        clauses = ["namespace = :ns"]
        params: dict[str, Any] = {"ns": namespace}
        if filter:
            clauses.append("metadata @> CAST(:flt AS jsonb)")
            params["flt"] = json.dumps(dict(filter))
        return " AND ".join(clauses), params

    def fetch(self, id: str, *, namespace: str = "") -> Metadata | None:
        stmt = text("SELECT metadata FROM embeddings WHERE vector_id = :id AND namespace = :ns")
        with self._engine.connect() as conn:
            row = conn.execute(stmt, {"id": id, "ns": namespace}).first()
        return dict(row[0]) if row is not None else None

    def delete(
        self,
        ids: list[str] | None = None,
        *,
        namespace: str = "",
        delete_all: bool = False,
    ) -> int:
        with self._engine.begin() as conn:
            if delete_all:
                result = conn.execute(
                    text("DELETE FROM embeddings WHERE namespace = :ns"), {"ns": namespace}
                )
                return result.rowcount
            if not ids:
                return 0
            stmt = text(
                "DELETE FROM embeddings WHERE namespace = :ns AND vector_id IN :ids"
            ).bindparams(bindparam("ids", expanding=True))
            result = conn.execute(stmt, {"ns": namespace, "ids": ids})
        return result.rowcount

    def count(self, *, namespace: str = "") -> int:
        with self._engine.connect() as conn:
            n = conn.execute(
                text("SELECT count(*) FROM embeddings WHERE namespace = :ns"), {"ns": namespace}
            ).scalar_one()
        return int(n)
