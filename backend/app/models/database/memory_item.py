from datetime import datetime

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.constants.memory_status import MemoryStatus
from app.constants.memory_type import MemoryType
from app.core.database import Base


class MemoryItem(Base):
    """从单个 Conversation 中提炼出的可持久化 Memory。

    v2.4-B6 第一版把 ``source_conversation_id`` 同时作为 Memory namespace
    与 provenance 根边界；不跨 Conversation 自动共享。消息起止 ID 保留可追溯
    来源，B7/B8 只能在这个持久化 Contract 上做抽取和检索。
    """

    __tablename__ = "memory_items"
    __table_args__ = (
        CheckConstraint(
            "memory_type IN ('fact', 'preference', 'constraint', 'decision', 'goal')",
            name="ck_memory_items_type",
        ),
        CheckConstraint(
            "status IN ('active', 'superseded', 'forgotten')",
            name="ck_memory_items_status",
        ),
        CheckConstraint(
            "source_message_start_id <= source_message_end_id",
            name="ck_memory_items_source_message_range",
        ),
        Index(
            "ix_memory_items_conversation_status",
            "source_conversation_id",
            "status",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source_conversation_id: Mapped[int] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    source_message_start_id: Mapped[int] = mapped_column(
        ForeignKey("conversation_messages.id", ondelete="CASCADE"),
        nullable=False,
    )
    source_message_end_id: Mapped[int] = mapped_column(
        ForeignKey("conversation_messages.id", ondelete="CASCADE"),
        nullable=False,
    )
    memory_type: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
    )
    content: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )
    status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default=MemoryStatus.ACTIVE.value,
        server_default=MemoryStatus.ACTIVE.value,
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
