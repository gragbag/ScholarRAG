"""add documents.collection (folders)

Revision ID: a1b2c3d4e5f6
Revises: 3c3e6004b87e
Create Date: 2026-07-25 00:00:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "a1b2c3d4e5f6"
down_revision: str | None = "3c3e6004b87e"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # server_default backfills existing rows to "default" without a rewrite; the
    # model-level default handles new inserts.
    op.add_column(
        "documents",
        sa.Column(
            "collection",
            sa.String(length=64),
            nullable=False,
            server_default="default",
        ),
    )
    op.create_index("ix_documents_collection", "documents", ["collection"])


def downgrade() -> None:
    op.drop_index("ix_documents_collection", table_name="documents")
    op.drop_column("documents", "collection")
