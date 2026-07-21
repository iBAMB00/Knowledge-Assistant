from datetime import datetime
from typing import Any
from sqlalchemy import func

from sqlalchemy import (
    DateTime,
    ForeignKey,
    Integer,
    Text,
    String,
    JSON,
)

from sqlalchemy.orm import (
    Mapped,
    mapped_column,
    relationship,
)

from app.core.database import Base


class DocumentChunk(Base):
    """
    文档切片模型。

    保存文档经过 ChunkService
    处理后的文本块。

    不负责切片算法。
    """

    __tablename__ = "document_chunks"


    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        comment="切片ID",
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


    chunk_index: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        comment="切片索引，从0开始",
    )


    content: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment="切片内容",
    )


    token_count: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        comment="当前切片包含的token数量",
       )


    chunk_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="recursive",
        comment="切片类型，递归或非递归切片",
    )


    chunk_metadata: Mapped[dict[str, Any] | None] = mapped_column(
        JSON,
        nullable=True,
        comment="元数据，用于存储切片相关的额外信息，JSON格式",
    )


    parent_chunk_id: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        comment="父切片ID",
    )


    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=func.now(),
        comment="创建/更新时间",
    )

    document: Mapped["Document"] = relationship(
        "Document",
        back_populates="chunks",
    )