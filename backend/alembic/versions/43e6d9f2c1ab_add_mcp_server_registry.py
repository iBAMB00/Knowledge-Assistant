"""add mcp server registry

Revision ID: 43e6d9f2c1ab
Revises: 7c4d91a2e5b8
Create Date: 2026-08-21 23:59:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "43e6d9f2c1ab"
down_revision: Union[str, Sequence[str], None] = "7c4d91a2e5b8"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "mcp_servers",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("server_id", sa.String(length=32), nullable=False),
        sa.Column(
            "transport",
            sa.String(length=32),
            server_default="stdio",
            nullable=False,
        ),
        sa.Column("command", sa.String(length=512), nullable=True),
        sa.Column("args", sa.JSON(), nullable=False),
        sa.Column(
            "timeout_seconds",
            sa.Float(),
            server_default="30",
            nullable=False,
        ),
        sa.Column(
            "enabled",
            sa.Boolean(),
            server_default=sa.true(),
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
            "timeout_seconds > 0 AND timeout_seconds <= 300",
            name="ck_mcp_servers_timeout_seconds",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("server_id", name="uq_mcp_servers_server_id"),
    )
    op.create_index(
        op.f("ix_mcp_servers_enabled"),
        "mcp_servers",
        ["enabled"],
        unique=False,
    )
    op.create_index(
        op.f("ix_mcp_servers_server_id"),
        "mcp_servers",
        ["server_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_mcp_servers_server_id"), table_name="mcp_servers")
    op.drop_index(op.f("ix_mcp_servers_enabled"), table_name="mcp_servers")
    op.drop_table("mcp_servers")
