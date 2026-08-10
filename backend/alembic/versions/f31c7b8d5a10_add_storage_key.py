"""add storage key for pluggable object storage

Revision ID: f31c7b8d5a10
Revises: e7f3b9a41c22
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "f31c7b8d5a10"
down_revision: Union[str, Sequence[str], None] = "e7f3b9a41c22"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """新增稳定 storage_key，并把历史 stored_name 回填为对象 key。"""
    with op.batch_alter_table("documents") as batch_op:
        batch_op.add_column(
            sa.Column(
                "storage_key",
                sa.String(length=500),
                nullable=True,
            )
        )
        batch_op.alter_column(
            "stored_name",
            existing_type=sa.String(length=255),
            nullable=True,
        )
        batch_op.alter_column(
            "path",
            existing_type=sa.String(length=500),
            nullable=True,
        )

    bind = op.get_bind()
    documents = sa.table(
        "documents",
        sa.column("id", sa.Integer()),
        sa.column("stored_name", sa.String()),
        sa.column("storage_key", sa.String()),
    )

    rows = bind.execute(
        sa.select(documents.c.id, documents.c.stored_name)
    ).all()

    for row in rows:
        if not row.stored_name:
            raise RuntimeError(
                f"document {row.id} has no stored_name for storage_key backfill"
            )
        bind.execute(
            documents.update()
            .where(documents.c.id == row.id)
            .values(storage_key=row.stored_name)
        )

    with op.batch_alter_table("documents") as batch_op:
        batch_op.alter_column(
            "storage_key",
            existing_type=sa.String(length=500),
            nullable=False,
        )
        batch_op.create_index(
            "ix_documents_storage_key",
            ["storage_key"],
            unique=True,
        )


def downgrade() -> None:
    """回退 storage_key；为 v0.17-C 后新增记录恢复 legacy 非空字段。"""
    bind = op.get_bind()
    documents = sa.table(
        "documents",
        sa.column("id", sa.Integer()),
        sa.column("stored_name", sa.String()),
        sa.column("path", sa.String()),
        sa.column("storage_key", sa.String()),
    )

    rows = bind.execute(
        sa.select(
            documents.c.id,
            documents.c.stored_name,
            documents.c.path,
            documents.c.storage_key,
        )
    ).all()

    for row in rows:
        values: dict[str, str] = {}
        if not row.stored_name:
            values["stored_name"] = f"legacy_{row.id}"
        if not row.path:
            values["path"] = row.storage_key or f"legacy_{row.id}"
        if values:
            bind.execute(
                documents.update()
                .where(documents.c.id == row.id)
                .values(**values)
            )

    with op.batch_alter_table("documents") as batch_op:
        batch_op.drop_index("ix_documents_storage_key")
        batch_op.drop_column("storage_key")
        batch_op.alter_column(
            "stored_name",
            existing_type=sa.String(length=255),
            nullable=False,
        )
        batch_op.alter_column(
            "path",
            existing_type=sa.String(length=500),
            nullable=False,
        )
