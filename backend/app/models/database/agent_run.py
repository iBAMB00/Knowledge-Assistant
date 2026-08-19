from datetime import datetime

from sqlalchemy import CheckConstraint, DateTime, ForeignKey, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column

from app.constants.agent_run_status import AgentRunStatus
from app.core.database import Base


class AgentRun(Base):
    """
    一次 Agent 执行的持久化运行事实。

    C1 记录生命周期、模型身份和安全关联信息；v2.0-E 起额外固化
    Runtime / Prompt / Toolset / Retrieval 与可选 Eval 版本快照。
    始终不保存完整 Prompt、隐藏推理、Tool 参数或 Tool Result 正文。
    """

    __tablename__ = "agent_runs"
    __table_args__ = (
        CheckConstraint(
            "status IN ('running', 'succeeded', 'failed')",
            name="ck_agent_runs_status",
        ),
        CheckConstraint(
            "tool_call_count >= 0",
            name="ck_agent_runs_tool_call_count",
        ),
    )

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
    )

    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )

    knowledge_base_id: Mapped[int] = mapped_column(
        ForeignKey("knowledge_bases.id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )

    request_id: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
        index=True,
    )

    status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default=AgentRunStatus.RUNNING.value,
        server_default=AgentRunStatus.RUNNING.value,
        index=True,
    )

    model_provider: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
    )

    model_name: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
    )

    # v2.0-E 起，新 Run 固化执行时版本；历史 Run 允许为空。
    agent_version: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
    )

    prompt_version: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
    )

    toolset_version: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
    )

    retrieval_config_version: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
    )

    eval_dataset_version: Mapped[str | None] = mapped_column(
        String(50),
        nullable=True,
    )

    evaluator_version: Mapped[str | None] = mapped_column(
        String(50),
        nullable=True,
    )

    tool_call_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default="0",
    )

    error_type: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
    )

    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    finished_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
