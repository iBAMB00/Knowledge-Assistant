"""extend agent run statuses for stateful runtime

Revision ID: d6f4a8c2b913
Revises: 8b7d3c4e2a91
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "d6f4a8c2b913"
down_revision: Union[str, Sequence[str], None] = "8b7d3c4e2a91"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """允许 Stateful execution attempt 记录 interrupted / cancelled。"""

    with op.batch_alter_table("agent_runs") as batch_op:
        batch_op.drop_constraint(
            "ck_agent_runs_status",
            type_="check",
        )
        batch_op.create_check_constraint(
            "ck_agent_runs_status",
            "status IN ('running', 'succeeded', 'failed', "
            "'interrupted', 'cancelled')",
        )


def downgrade() -> None:
    """回退前把新状态归一为 failed，再恢复旧约束。"""

    op.execute(
        sa.text(
            "UPDATE agent_runs SET status='failed' "
            "WHERE status IN ('interrupted', 'cancelled')"
        )
    )
    with op.batch_alter_table("agent_runs") as batch_op:
        batch_op.drop_constraint(
            "ck_agent_runs_status",
            type_="check",
        )
        batch_op.create_check_constraint(
            "ck_agent_runs_status",
            "status IN ('running', 'succeeded', 'failed')",
        )
