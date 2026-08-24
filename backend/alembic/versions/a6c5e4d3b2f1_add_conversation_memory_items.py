"""add conversation memory items

Revision ID: a6c5e4d3b2f1
Revises: f4a1c9d8e2b7
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "a6c5e4d3b2f1"
down_revision: Union[str, Sequence[str], None] = "f4a1c9d8e2b7"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """创建 Conversation-derived Memory 持久化结构。"""

    op.create_table(
        "memory_items",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("source_conversation_id", sa.Integer(), nullable=False),
        sa.Column("source_message_start_id", sa.Integer(), nullable=False),
        sa.Column("source_message_end_id", sa.Integer(), nullable=False),
        sa.Column("memory_type", sa.String(length=32), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column(
            "status",
            sa.String(length=20),
            server_default="active",
            nullable=False,
        ),
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
            "memory_type IN ('fact', 'preference', 'constraint', 'decision', 'goal')",
            name="ck_memory_items_type",
        ),
        sa.CheckConstraint(
            "status IN ('active', 'superseded', 'forgotten')",
            name="ck_memory_items_status",
        ),
        sa.CheckConstraint(
            "source_message_start_id <= source_message_end_id",
            name="ck_memory_items_source_message_range",
        ),
        sa.ForeignKeyConstraint(
            ["source_conversation_id"],
            ["conversations.id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["source_message_start_id"],
            ["conversation_messages.id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["source_message_end_id"],
            ["conversation_messages.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_memory_items_source_conversation_id",
        "memory_items",
        ["source_conversation_id"],
    )
    op.create_index(
        "ix_memory_items_conversation_status",
        "memory_items",
        ["source_conversation_id", "status"],
    )


def downgrade() -> None:
    """移除 Conversation-derived Memory 持久化结构。"""

    op.drop_index(
        "ix_memory_items_conversation_status",
        table_name="memory_items",
    )
    op.drop_index(
        "ix_memory_items_source_conversation_id",
        table_name="memory_items",
    )
    op.drop_table("memory_items")
