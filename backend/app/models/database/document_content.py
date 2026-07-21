from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


if TYPE_CHECKING:
    from app.models.database.document import Document
    from app.models.database.document_chunk import DocumentChunk



class DocumentContent(Base):
    """
    文档解析内容数据库模型。

    保存文档经过解析后的完整文本。
    同一个文档可以存在多个解析版本。
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
        index=True,
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
        comment="解析器类型，如pdf_parser、markdown_parser等",
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

    version: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=1,
        server_default="1",
        comment="文档解析版本号",
    )

    parser_version: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        comment="解析器版本",
    )

    document: Mapped["Document"] = relationship(
        "Document",
        back_populates="contents",
    )

    chunks: Mapped[list["DocumentChunk"]] = relationship(
        "DocumentChunk",
        back_populates="document_content",
        cascade="all, delete-orphan",
        order_by="DocumentChunk.chunk_index",
    )