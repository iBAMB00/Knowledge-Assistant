"""add agent threads and durable checkpoints

Revision ID: 8b7d3c4e2a91
Revises: 5a9c1d7e3b42
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "8b7d3c4e2a91"
down_revision: Union[str, Sequence[str], None] = "5a9c1d7e3b42"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "agent_threads",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("thread_id", sa.String(length=128), nullable=False),
        sa.Column("conversation_id", sa.Integer(), nullable=False),
        sa.Column("state_schema_version", sa.String(length=16), nullable=False),
        sa.Column(
            "status",
            sa.String(length=20),
            server_default="ready",
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
            "status IN ('ready', 'running', 'waiting', 'succeeded', "
            "'failed', 'cancelled')",
            name="ck_agent_threads_status",
        ),
        sa.ForeignKeyConstraint(
            ["conversation_id"],
            ["conversations.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("conversation_id"),
        sa.UniqueConstraint("thread_id"),
    )
    op.create_index(
        "ix_agent_threads_thread_id",
        "agent_threads",
        ["thread_id"],
        unique=True,
    )
    op.create_index(
        "ix_agent_threads_conversation_id",
        "agent_threads",
        ["conversation_id"],
        unique=True,
    )
    op.create_index(
        "ix_agent_threads_status",
        "agent_threads",
        ["status"],
    )

    op.create_table(
        "agent_checkpoints",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("agent_thread_id", sa.Integer(), nullable=False),
        sa.Column("sequence", sa.Integer(), nullable=False),
        sa.Column(
            "checkpoint_schema_version",
            sa.String(length=16),
            nullable=False,
        ),
        sa.Column(
            "state_schema_version",
            sa.String(length=16),
            nullable=False,
        ),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["agent_thread_id"],
            ["agent_threads.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "agent_thread_id",
            "sequence",
            name="uq_agent_checkpoints_thread_sequence",
        ),
    )
    op.create_index(
        "ix_agent_checkpoints_agent_thread_id",
        "agent_checkpoints",
        ["agent_thread_id"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_agent_checkpoints_agent_thread_id",
        table_name="agent_checkpoints",
    )
    op.drop_table("agent_checkpoints")

    op.drop_index("ix_agent_threads_status", table_name="agent_threads")
    op.drop_index(
        "ix_agent_threads_conversation_id",
        table_name="agent_threads",
    )
    op.drop_index("ix_agent_threads_thread_id", table_name="agent_threads")
    op.drop_table("agent_threads")
