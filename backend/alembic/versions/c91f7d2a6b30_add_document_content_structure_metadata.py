"""add document content structure metadata

Revision ID: c91f7d2a6b30
Revises: b4c1f0d2e8a7
Create Date: 2026-08-09
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "c91f7d2a6b30"
down_revision: Union[str, Sequence[str], None] = "b4c1f0d2e8a7"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "document_contents",
        sa.Column(
            "structure_metadata",
            sa.JSON(),
            nullable=True,
        ),
    )


def downgrade() -> None:
    op.drop_column(
        "document_contents",
        "structure_metadata",
    )
