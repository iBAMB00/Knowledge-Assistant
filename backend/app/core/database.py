from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker


DATABASE_URL = "sqlite:///./secure_assistant.db"


# 创建数据库引擎。
# SQLite 会将数据保存到项目目录下的 secure_assistant.db 文件。
engine = create_engine(
    DATABASE_URL,
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