from __future__ import annotations

from datetime import datetime

from sqlalchemy import CheckConstraint, DateTime, ForeignKey, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.constants.agent_state_status import AgentStateStatus
from app.core.database import Base


class AgentThread(Base):
    """一个 Agent Conversation 对应的持久化 Stateful Runtime 线程。"""

    __tablename__ = "agent_threads"
    __table_args__ = (
        CheckConstraint(
            "status IN ('ready', 'running', 'waiting', 'succeeded', "
            "'failed', 'cancelled')",
            name="ck_agent_threads_status",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    thread_id: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
        unique=True,
        index=True,
    )
    conversation_id: Mapped[int] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True,
    )
    state_schema_version: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
    )
    status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default=AgentStateStatus.READY.value,
        server_default=AgentStateStatus.READY.value,
        index=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    checkpoints: Mapped[list["AgentCheckpoint"]] = relationship(
        "AgentCheckpoint",
        back_populates="thread",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="AgentCheckpoint.sequence",
    )
