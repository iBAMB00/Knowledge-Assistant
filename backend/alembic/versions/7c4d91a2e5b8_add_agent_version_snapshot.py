"""add agent run version snapshot

Revision ID: 7c4d91a2e5b8
Revises: 2d7b4a9c1e60
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "7c4d91a2e5b8"
down_revision: Union[str, Sequence[str], None] = "2d7b4a9c1e60"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """为 AgentRun 增加可回溯的执行版本快照。"""

    op.add_column(
        "agent_runs",
        sa.Column("agent_version", sa.String(length=64), nullable=True),
    )
    op.add_column(
        "agent_runs",
        sa.Column("prompt_version", sa.String(length=64), nullable=True),
    )
    op.add_column(
        "agent_runs",
        sa.Column("toolset_version", sa.String(length=64), nullable=True),
    )
    op.add_column(
        "agent_runs",
        sa.Column(
            "retrieval_config_version",
            sa.String(length=64),
            nullable=True,
        ),
    )
    op.add_column(
        "agent_runs",
        sa.Column("eval_dataset_version", sa.String(length=50), nullable=True),
    )
    op.add_column(
        "agent_runs",
        sa.Column("evaluator_version", sa.String(length=50), nullable=True),
    )


def downgrade() -> None:
    """移除 AgentRun 版本快照字段。"""

    op.drop_column("agent_runs", "evaluator_version")
    op.drop_column("agent_runs", "eval_dataset_version")
    op.drop_column("agent_runs", "retrieval_config_version")
    op.drop_column("agent_runs", "toolset_version")
    op.drop_column("agent_runs", "prompt_version")
    op.drop_column("agent_runs", "agent_version")
