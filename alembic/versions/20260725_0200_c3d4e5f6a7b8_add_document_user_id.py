"""add documents.user_id (per-user ownership)

Revision ID: c3d4e5f6a7b8
Revises: b2c3d4e5f6a7
Create Date: 2026-07-25 02:00:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "c3d4e5f6a7b8"
down_revision: str | None = "b2c3d4e5f6a7"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Nullable: existing rows become NULL (public/seed), no rewrite needed.
    op.add_column("documents", sa.Column("user_id", sa.UUID(), nullable=True))
    op.create_index("ix_documents_user_id", "documents", ["user_id"])
    op.create_foreign_key(
        "fk_documents_user_id",
        "documents",
        "users",
        ["user_id"],
        ["id"],
        ondelete="SET NULL",
    )


def downgrade() -> None:
    op.drop_constraint("fk_documents_user_id", "documents", type_="foreignkey")
    op.drop_index("ix_documents_user_id", table_name="documents")
    op.drop_column("documents", "user_id")
