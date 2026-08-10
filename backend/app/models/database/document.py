from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, ForeignKey, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.constants.document_status import DocumentStatus
from app.core.database import Base


if TYPE_CHECKING:
    from app.models.database.document_content import DocumentContent
    from app.models.database.knowledge_base import KnowledgeBase


class Document(Base):
    """
    文档数据库模型。

    保存上传文档的元数据和生命周期状态，
    不保存文档正文及解析后的文本内容。
    """

    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
    )

    knowledge_base_id: Mapped[int | None] = mapped_column(
        ForeignKey(
            "knowledge_bases.id",
            ondelete="RESTRICT",
        ),
        nullable=True,
        index=True,
        comment="所属知识库ID；NULL仅用于兼容历史数据",
    )

    filename: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        comment="文档文件名",
    )

    storage_key: Mapped[str] = mapped_column(
        String(500),
        nullable=False,
        unique=True,
        index=True,
        comment="与底层存储实现无关的稳定对象Key",
    )

    # Legacy compatibility：v0.17-C 起业务逻辑不再依赖这两个字段。
    stored_name: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        comment="历史本地存储文件名，待后续迁移删除",
    )

    path: Mapped[str | None] = mapped_column(
        String(500),
        nullable=True,
        comment="历史本地绝对/相对路径，待后续迁移删除",
    )

    size: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    status: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default=DocumentStatus.UPLOADED.value,
        server_default=DocumentStatus.UPLOADED.value,
        comment="文档状态，上传中、解析中、已解析、解析失败",
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        comment="文档上传时间",
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
        comment="文档最后更新时间",
    )

    knowledge_base: Mapped["KnowledgeBase | None"] = relationship(
        "KnowledgeBase",
        back_populates="documents",
    )

    contents: Mapped[list["DocumentContent"]] = relationship(
        "DocumentContent",
        back_populates="document",
        cascade="all, delete-orphan",
        order_by="DocumentContent.created_at",
    )