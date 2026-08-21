from datetime import datetime

from sqlalchemy import Boolean, DateTime, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class MCPServer(Base):
    """MCP Server 持久化配置。"""

    __tablename__ = "mcp_servers"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
    )

    server_id: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        unique=True,
        index=True,
    )

    transport: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="stdio",
    )

    command: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    args_json: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        default="[]",
    )

    timeout_seconds: Mapped[float] = mapped_column(
        nullable=False,
        default=30,
    )

    enabled: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
    )

    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="registered",
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
