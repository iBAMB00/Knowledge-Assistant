"""update processing job type check

Revision ID: 9ac29a252355
Revises: e6f3637f8a6f
Create Date: 2026-08-02 21:11:50.330650

"""
from typing import Sequence, Union

from alembic import op


revision: str = "9ac29a252355"
down_revision: Union[str, Sequence[str], None] = "e6f3637f8a6f"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """允许processing_jobs使用full_pipeline任务类型。"""

    with op.batch_alter_table(
        "processing_jobs",
        recreate="always",
    ) as batch_op:
        batch_op.drop_constraint(
            "ck_processing_jobs_job_type",
            type_="check",
        )

        batch_op.create_check_constraint(
            "ck_processing_jobs_job_type",
            "job_type IN ("
            "'document_processing', "
            "'embedding', "
            "'full_pipeline'"
            ")",
        )


def downgrade() -> None:
    """恢复旧的任务类型约束。"""

    with op.batch_alter_table(
        "processing_jobs",
        recreate="always",
    ) as batch_op:
        batch_op.drop_constraint(
            "ck_processing_jobs_job_type",
            type_="check",
        )

        batch_op.create_check_constraint(
            "ck_processing_jobs_job_type",
            "job_type IN ("
            "'document_processing', "
            "'embedding'"
            ")",
        )