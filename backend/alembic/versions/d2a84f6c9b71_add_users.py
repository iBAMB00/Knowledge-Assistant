"""add users

Revision ID: d2a84f6c9b71
Revises: c91f7d2a6b30
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "d2a84f6c9b71"
down_revision: Union[str, Sequence[str], None] = "c91f7d2a6b30"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """创建本地认证用户表。"""
    op.create_table(
        "users",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("email", sa.String(length=320), nullable=False),
        sa.Column("password_hash", sa.String(length=512), nullable=False),
        sa.Column(
            "is_active",
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
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_users_email",
        "users",
        ["email"],
        unique=True,
    )


def downgrade() -> None:
    """删除本地认证用户表。"""
    op.drop_index("ix_users_email", table_name="users")
    op.drop_table("users")
