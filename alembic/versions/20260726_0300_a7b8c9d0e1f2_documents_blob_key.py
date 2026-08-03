"""documents: replace raw_content (bytea) with blob_key (object storage)

Raw PDF bytes move out of Postgres into a blob store (R2/S3/local); the row keeps
only a short key. Existing rows lose their raw bytes (they can't be reprocessed),
which is fine — they're already ingested.

Revision ID: a7b8c9d0e1f2
Revises: f6a7b8c9d0e1
Create Date: 2026-07-26 03:00:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "a7b8c9d0e1f2"
down_revision: str | None = "f6a7b8c9d0e1"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("documents", sa.Column("blob_key", sa.String(length=256), nullable=True))
    op.drop_column("documents", "raw_content")


def downgrade() -> None:
    op.add_column("documents", sa.Column("raw_content", sa.LargeBinary(), nullable=True))
    op.drop_column("documents", "blob_key")
