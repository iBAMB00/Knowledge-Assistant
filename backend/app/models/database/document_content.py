from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Integer, Text, String
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class DocumentContent(Base):
    """
    文档解析内容模型。

    保存文档解析后的文本内容。
    """

    __tablename__ = "document_contents"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        index=True,
    )

    document_id: Mapped[int] = mapped_column(
        ForeignKey("documents.id"),
        nullable=False,
        unique=True,
    )

    content: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    parser_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
    )