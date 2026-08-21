from datetime import datetime

from sqlalchemy import Boolean, CheckConstraint, DateTime, Float, Integer, JSON, String, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class MCPServer(Base):
    """MCP Server 的持久化配置事实。

    这里只保存可恢复的期望配置，不保存连接健康状态等运行时观察值。
    Runtime 仍使用 ``MCPServerConfig``，避免 SQLAlchemy Model 进入 Agent 核心。
    """

    __tablename__ = "mcp_servers"
    __table_args__ = (
        UniqueConstraint("server_id", name="uq_mcp_servers_server_id"),
        CheckConstraint(
            "timeout_seconds > 0 AND timeout_seconds <= 300",
            name="ck_mcp_servers_timeout_seconds",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    server_id: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        index=True,
    )
    transport: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="stdio",
        server_default="stdio",
    )
    command: Mapped[str | None] = mapped_column(
        String(512),
        nullable=True,
    )
    args: Mapped[list[str]] = mapped_column(
        JSON,
        nullable=False,
        default=list,
    )
    timeout_seconds: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=30.0,
        server_default="30",
    )
    enabled: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
        server_default="1",
        index=True,
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
