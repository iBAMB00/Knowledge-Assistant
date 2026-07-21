from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import (
    DateTime,
    ForeignKey,
    Integer,
    JSON,
    String,
    Text,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


if TYPE_CHECKING:
    from app.models.database.document_chunk import DocumentChunk


class ChunkEmbedding(Base):
    """
    文档切片向量模型。

    保存 Chunk 对应的向量化结果。

    不负责：
    - 向量生成
    - 向量检索
    """

    __tablename__ = "chunk_embeddings"


    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        comment="向量记录ID",
    )


    document_chunk_id: Mapped[int] = mapped_column(
        ForeignKey(
            "document_chunks.id",
            ondelete="CASCADE",
        ),
        nullable=False,
        unique=True,
        comment="关联切片ID",
    )


    vector: Mapped[list[float] | None] = mapped_column(
        JSON,
        nullable=True,
        comment="文本向量数据",
    )


    embedding_model: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        comment="Embedding模型名称",
    )


    embedding_dimension: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        comment="向量维度",
    )


    embedding_metadata: Mapped[dict[str, Any] | None] = mapped_column(
        JSON,
        nullable=True,
        comment="向量额外元数据",
    )


    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        comment="向量创建时间",
    )


    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
        comment="向量更新时间",
    )


    chunk: Mapped["DocumentChunk"] = relationship(
        "DocumentChunk",
        back_populates="embedding",
    )