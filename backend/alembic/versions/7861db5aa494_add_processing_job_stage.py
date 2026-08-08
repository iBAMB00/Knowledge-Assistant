"""add processing job stage

Revision ID: 7861db5aa494
Revises: 9ac29a252355
Create Date: 2026-08-06 02:02:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "7861db5aa494"
down_revision: Union[str, Sequence[str], None] = "9ac29a252355"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """增加处理任务业务阶段字段和数据库约束。"""

    with op.batch_alter_table(
        "processing_jobs",
        recreate="always",
    ) as batch_op:
        batch_op.add_column(
            sa.Column(
                "stage",
                sa.String(length=20),
                nullable=False,
                server_default="queued",
            )
        )

        batch_op.create_check_constraint(
            "ck_processing_jobs_stage",
            (
                "stage IN ("
                "'queued', "
                "'parsing', "
                "'chunking', "
                "'embedding', "
                "'indexing', "
                "'finalizing', "
                "'completed'"
                ")"
            ),
        )


def downgrade() -> None:
    """删除处理任务业务阶段字段。"""

    with op.batch_alter_table(
        "processing_jobs",
        recreate="always",
    ) as batch_op:
        batch_op.drop_constraint(
            "ck_processing_jobs_stage",
            type_="check",
        )
        batch_op.drop_column("stage")
