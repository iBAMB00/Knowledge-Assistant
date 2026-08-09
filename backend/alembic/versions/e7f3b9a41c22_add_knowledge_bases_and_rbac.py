"""add knowledge bases and basic rbac

Revision ID: e7f3b9a41c22
Revises: d2a84f6c9b71
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "e7f3b9a41c22"
down_revision: Union[str, Sequence[str], None] = "d2a84f6c9b71"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """增加用户角色、知识库实体和文档知识库归属。"""
    with op.batch_alter_table("users") as batch_op:
        batch_op.add_column(
            sa.Column(
                "role",
                sa.String(length=20),
                server_default="user",
                nullable=False,
            )
        )
        batch_op.create_check_constraint(
            "ck_users_role",
            "role IN ('user', 'admin')",
        )

    op.create_table(
        "knowledge_bases",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("owner_id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=100), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
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
        sa.ForeignKeyConstraint(["owner_id"], ["users.id"], ondelete="RESTRICT"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("owner_id", "name", name="uq_knowledge_bases_owner_name"),
    )
    op.create_index("ix_knowledge_bases_owner_id", "knowledge_bases", ["owner_id"])

    # 兼容已有评估/历史文档：旧记录暂时保持 NULL，不猜测其所有者。
    with op.batch_alter_table("documents") as batch_op:
        batch_op.add_column(sa.Column("knowledge_base_id", sa.Integer(), nullable=True))
        batch_op.create_foreign_key(
            "fk_documents_knowledge_base_id",
            "knowledge_bases",
            ["knowledge_base_id"],
            ["id"],
            ondelete="RESTRICT",
        )
        batch_op.create_index("ix_documents_knowledge_base_id", ["knowledge_base_id"])


def downgrade() -> None:
    """移除知识库与最小 RBAC 结构。"""
    with op.batch_alter_table("documents") as batch_op:
        batch_op.drop_index("ix_documents_knowledge_base_id")
        batch_op.drop_constraint("fk_documents_knowledge_base_id", type_="foreignkey")
        batch_op.drop_column("knowledge_base_id")

    op.drop_index("ix_knowledge_bases_owner_id", table_name="knowledge_bases")
    op.drop_table("knowledge_bases")

    with op.batch_alter_table("users") as batch_op:
        batch_op.drop_constraint("ck_users_role", type_="check")
        batch_op.drop_column("role")
