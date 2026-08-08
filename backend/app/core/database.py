from collections.abc import Generator
import sqlite3

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session, declarative_base, sessionmaker

from app.core.config import get_settings


settings = get_settings()


def enable_sqlite_foreign_keys(dbapi_connection, connection_record) -> None:
    """为每个 SQLite 数据库连接开启外键约束。"""
    if not isinstance(dbapi_connection, sqlite3.Connection):
        return

    cursor = dbapi_connection.cursor()
    try:
        cursor.execute("PRAGMA foreign_keys=ON")
    finally:
        cursor.close()


def build_engine_options(database_url: str) -> dict:
    """根据数据库方言生成 SQLAlchemy Engine 参数。"""
    backend = make_url(database_url).get_backend_name()
    options: dict = {"pool_pre_ping": settings.database_pool_pre_ping}

    if backend == "sqlite":
        options["connect_args"] = {"check_same_thread": False}
    else:
        options["pool_recycle"] = settings.database_pool_recycle

    return options


def create_database_engine(database_url: str) -> Engine:
    """创建支持 SQLite 与 PostgreSQL 的数据库 Engine。"""
    engine = create_engine(database_url, **build_engine_options(database_url))

    if make_url(database_url).get_backend_name() == "sqlite":
        event.listen(engine, "connect", enable_sqlite_foreign_keys)

    return engine


engine = create_database_engine(settings.DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db() -> Generator[Session, None, None]:
    """获取数据库 Session，用于 FastAPI 依赖注入。"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
