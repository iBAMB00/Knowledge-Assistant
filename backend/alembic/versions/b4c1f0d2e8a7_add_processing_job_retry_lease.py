"""add processing job retry lease

Revision ID: b4c1f0d2e8a7
Revises: 7861db5aa494
Create Date: 2026-08-09
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "b4c1f0d2e8a7"
down_revision: Union[str, Sequence[str], None] = "7861db5aa494"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "processing_jobs",
        sa.Column(
            "attempt_count",
            sa.Integer(),
            nullable=False,
            server_default="0",
        ),
    )
    op.add_column(
        "processing_jobs",
        sa.Column("lease_expires_at", sa.DateTime(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("processing_jobs", "lease_expires_at")
    op.drop_column("processing_jobs", "attempt_count")
