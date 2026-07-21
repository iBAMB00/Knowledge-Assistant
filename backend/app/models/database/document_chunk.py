from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

from app.constants.embedding_status import EmbeddingStatus

from sqlalchemy import (
    DateTime,
    ForeignKey,
    Integer,
    Text,
    String,
    JSON,
    UniqueConstraint,
    func,
)

from sqlalchemy.orm import (
    Mapped,
    mapped_column,
    relationship,
)

from app.core.database import Base


if TYPE_CHECKING:
    from app.models.database.document_content import DocumentContent


class DocumentChunk(Base):
    """
    文档切片模型。

    保存文档解析内容经过 ChunkService
    处理后的文本块。

    不负责切片算法。
    """

    __tablename__ = "document_chunks"

    __table_args__ = (
        UniqueConstraint(
            "document_content_id",
            "chunk_index",
            name="uq_document_content_chunk_index",
        ),
    )

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        comment="切片ID",
    )

    document_content_id: Mapped[int] = mapped_column(
        ForeignKey(
            "document_contents.id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
        comment="关联解析内容ID",
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

    chunk_strategy: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="recursive_character",
        comment="切片策略名称",
    )
    
    embedding_status: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default=EmbeddingStatus.PENDING.value,
        server_default=EmbeddingStatus.PENDING.value,
        comment="向量化状态",
    )

    chunk_metadata: Mapped[dict[str, Any] | None] = mapped_column(
        JSON,
        nullable=True,
        comment="切片相关元数据",
    )

    parent_chunk_id: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        comment="父切片ID",
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        comment="切片创建时间",
    )

    document_content: Mapped["DocumentContent"] = relationship(
        "DocumentContent",
        back_populates="chunks",
    )