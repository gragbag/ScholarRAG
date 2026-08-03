"""add embeddings table (pgvector) for the consolidated vector store

Replaces Pinecone: vectors now live in Postgres via pgvector, alongside the
documents/BM25 data. HNSW index for cosine KNN; GIN index for the {collection,
owner} JSONB containment filter.

Revision ID: f6a7b8c9d0e1
Revises: e5f6a7b8c9d0
Create Date: 2026-07-26 02:00:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "f6a7b8c9d0e1"
down_revision: str | None = "e5f6a7b8c9d0"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute(
        """
        CREATE TABLE embeddings (
            vector_id  varchar(128) PRIMARY KEY,
            namespace  varchar(64)  NOT NULL DEFAULT '',
            embedding  vector(384)  NOT NULL,
            metadata   jsonb        NOT NULL DEFAULT '{}'
        )
        """
    )
    # Approximate nearest-neighbour index for fast cosine KNN (`<=>`).
    op.execute("CREATE INDEX ix_embeddings_hnsw ON embeddings USING hnsw (embedding vector_cosine_ops)")
    # Containment (`@>`) filter on {collection, owner}.
    op.execute("CREATE INDEX ix_embeddings_metadata ON embeddings USING gin (metadata)")


def downgrade() -> None:
    op.drop_table("embeddings")
    # Leave the `vector` extension installed — cheap, and other objects may use it.
