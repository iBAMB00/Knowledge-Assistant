from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import Boolean, CheckConstraint, DateTime, Integer, String, func, true
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.constants.user_role import UserRole
from app.core.database import Base

if TYPE_CHECKING:
    from app.models.database.knowledge_base import KnowledgeBase


class User(Base):
    """
    用户数据库模型。

    保存认证所需字段与最小 user/admin 角色。
    具体资源访问边界由 KnowledgeBase ownership 决定。
    """

    __tablename__ = "users"
    __table_args__ = (
        CheckConstraint(
            "role IN ('user', 'admin')",
            name="ck_users_role",
        ),
    )

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
    )

    email: Mapped[str] = mapped_column(
        String(320),
        nullable=False,
        unique=True,
        index=True,
        comment="规范化后的用户邮箱",
    )

    password_hash: Mapped[str] = mapped_column(
        String(512),
        nullable=False,
        comment="Argon2 密码哈希",
    )

    role: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default=UserRole.USER.value,
        server_default=UserRole.USER.value,
        comment="用户角色：user/admin",
    )

    is_active: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
        server_default=true(),
        comment="用户是否可登录和访问受保护资源",
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

    knowledge_bases: Mapped[list["KnowledgeBase"]] = relationship(
        "KnowledgeBase",
        back_populates="owner",
    )
