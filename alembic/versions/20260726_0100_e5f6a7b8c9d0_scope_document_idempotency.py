"""scope document idempotency to (user_id, collection, content_hash)

Idempotency was a GLOBAL unique on content_hash, which broke once documents
became per-user and per-folder: adding a paper already in the public seed (or in
another folder) deduped to that row instead of creating the user's own copy. Swap
the global unique index for a composite unique constraint on
(user_id, collection, content_hash).

Revision ID: e5f6a7b8c9d0
Revises: d4e5f6a7b8c9
Create Date: 2026-07-26 01:00:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "e5f6a7b8c9d0"
down_revision: str | None = "d4e5f6a7b8c9"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Drop the globally-unique index, recreate it as a plain (non-unique) lookup index.
    op.drop_index("ix_documents_content_hash", table_name="documents")
    op.create_index("ix_documents_content_hash", "documents", ["content_hash"])
    op.create_unique_constraint(
        "uq_documents_owner_collection_hash",
        "documents",
        ["user_id", "collection", "content_hash"],
    )


def downgrade() -> None:
    op.drop_constraint("uq_documents_owner_collection_hash", "documents", type_="unique")
    op.drop_index("ix_documents_content_hash", table_name="documents")
    op.create_index("ix_documents_content_hash", "documents", ["content_hash"], unique=True)
