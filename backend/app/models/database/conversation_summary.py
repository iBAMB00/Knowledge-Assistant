from datetime import datetime

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class ConversationSummary(Base):
    """Conversation 旧历史的增量压缩结果。

    Summary 是模型 Context 资产，不是 Agent State / Checkpoint / Memory。
    summarized_through_message_id 表示当前摘要已经连续覆盖到哪条消息。
    """

    __tablename__ = "conversation_summaries"
    __table_args__ = (
        UniqueConstraint(
            "conversation_id",
            name="uq_conversation_summaries_conversation_id",
        ),
        CheckConstraint(
            "summarized_through_message_id > 0",
            name="ck_conversation_summaries_message_boundary",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    conversation_id: Mapped[int] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    summarized_through_message_id: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )
    prompt_id: Mapped[str] = mapped_column(
        String(80),
        nullable=False,
    )
    prompt_version: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
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
