from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


if TYPE_CHECKING:
    from app.models.database.document import Document


class DocumentContent(Base):
    """
    文档解析内容数据库模型。

    保存文档解析后的完整文本。
    同一个文档只保留一条当前有效的解析结果。
    """

    __tablename__ = "document_contents"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        comment="文档解析内容ID",
    )

    document_id: Mapped[int] = mapped_column(
        ForeignKey(
            "documents.id",
            ondelete="CASCADE",
        ),
        nullable=False,
        unique=True,
        comment="关联的文档ID",
    )

    content: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment="文档解析后的完整文本",
    )

    parser_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        comment="解析器类型，如OpenAI、Google Cloud等",
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        comment="文档解析时间",
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
        comment="文档解析最后更新时间",
    )

    document: Mapped["Document"] = relationship(
        "Document",
        back_populates="parsed_content",
    )