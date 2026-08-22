from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, ForeignKey, Integer, JSON, String, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


class AgentCheckpoint(Base):
    """AgentThread 的不可变顺序状态快照。"""

    __tablename__ = "agent_checkpoints"
    __table_args__ = (
        UniqueConstraint(
            "agent_thread_id",
            "sequence",
            name="uq_agent_checkpoints_thread_sequence",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    agent_thread_id: Mapped[int] = mapped_column(
        ForeignKey("agent_threads.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    sequence: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )
    checkpoint_schema_version: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
    )
    state_schema_version: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
    )
    payload: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    thread: Mapped["AgentThread"] = relationship(
        "AgentThread",
        back_populates="checkpoints",
    )
