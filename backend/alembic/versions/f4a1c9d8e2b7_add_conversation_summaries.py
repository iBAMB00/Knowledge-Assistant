"""add conversation summaries

Revision ID: f4a1c9d8e2b7
Revises: d6f4a8c2b913
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "f4a1c9d8e2b7"
down_revision: Union[str, Sequence[str], None] = "d6f4a8c2b913"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """创建 Conversation 的单份增量 Summary 持久化结构。"""

    op.create_table(
        "conversation_summaries",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("conversation_id", sa.Integer(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column(
            "summarized_through_message_id",
            sa.Integer(),
            nullable=False,
        ),
        sa.Column("prompt_id", sa.String(length=80), nullable=False),
        sa.Column("prompt_version", sa.String(length=64), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.CheckConstraint(
            "summarized_through_message_id > 0",
            name="ck_conversation_summaries_message_boundary",
        ),
        sa.ForeignKeyConstraint(
            ["conversation_id"],
            ["conversations.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "conversation_id",
            name="uq_conversation_summaries_conversation_id",
        ),
    )
    op.create_index(
        "ix_conversation_summaries_conversation_id",
        "conversation_summaries",
        ["conversation_id"],
    )


def downgrade() -> None:
    """移除 Conversation Summary 持久化结构。"""

    op.drop_index(
        "ix_conversation_summaries_conversation_id",
        table_name="conversation_summaries",
    )
    op.drop_table("conversation_summaries")
