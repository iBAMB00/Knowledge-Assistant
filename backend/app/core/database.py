import sqlite3

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import declarative_base, sessionmaker

from app.core.config import get_settings


settings = get_settings()

@event.listens_for(Engine, "connect")
def enable_sqlite_foreign_keys(
    dbapi_connection,
    connection_record,
) -> None:
    """为每个 SQLite 数据库连接开启外键约束。"""

    if not isinstance(
        dbapi_connection,
        sqlite3.Connection,
    ):
        return

    cursor = dbapi_connection.cursor()

    try:
        cursor.execute(
            "PRAGMA foreign_keys=ON"
        )
    finally:
        cursor.close()

# 创建数据库引擎。
# SQLite 会将数据保存到项目目录下的 secure_assistant.db 文件。
engine = create_engine(
    settings.DATABASE_URL,
    connect_args={
        "check_same_thread": False,
    },
)


# 创建数据库会话工厂。
# 每次业务操作通过 Session 获取数据库连接。
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)


# 所有数据库模型继承这个 Base。
Base = declarative_base()


def get_db():
    """
    获取数据库 Session。

    用于 FastAPI Dependency Injection。
    """

    db = SessionLocal()

    try:
        yield db

    finally:
        db.close()