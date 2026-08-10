from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base

if TYPE_CHECKING:
    from app.models.database.document import Document
    from app.models.database.user import User


class KnowledgeBase(Base):
    """用户拥有的知识库业务容器。"""

    __tablename__ = "knowledge_bases"
    __table_args__ = (
        UniqueConstraint(
            "owner_id",
            "name",
            name="uq_knowledge_bases_owner_name",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    owner_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
        comment="知识库所有者用户ID",
    )
    name: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        comment="知识库名称",
    )
    description: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="知识库说明",
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

    owner: Mapped["User"] = relationship(
        "User",
        back_populates="knowledge_bases",
    )
    documents: Mapped[list["Document"]] = relationship(
        "Document",
        back_populates="knowledge_base",
    )
