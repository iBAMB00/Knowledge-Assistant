import os
from pathlib import Path
import subprocess

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.engine import make_url
from sqlalchemy.orm import sessionmaker

from app.core.config import get_settings
from app.core.database import enable_sqlite_foreign_keys

settings = get_settings()


@pytest.fixture(scope="session")
def test_engine():
    """
    创建测试数据库。

    使用Alembic初始化表结构，并显式控制
    SQLite事务，保证Savepoint正确参与外层事务。
    """

    test_database_url = make_url(
        settings.TEST_DATABASE_URL
    )

    if test_database_url.get_backend_name() != "sqlite":
        raise RuntimeError(
            "tests currently require a SQLite database"
        )

    database_name = test_database_url.database

    if not database_name:
        raise RuntimeError(
            "test database path is missing"
        )

    test_db_path = Path(database_name)

    if test_db_path.exists():
        test_db_path.unlink()


    subprocess.run(
        [
            "alembic",
            "upgrade",
            "head",
        ],
        env={
            **os.environ,
            "DATABASE_URL": settings.TEST_DATABASE_URL,
        },
        check=True,
    )


    engine = create_engine(
        settings.TEST_DATABASE_URL,
        connect_args={
            "check_same_thread": False,
        },
    )

    @event.listens_for(engine, "connect")
    def configure_sqlite_connection(
        dbapi_connection,
        connection_record,
    ) -> None:
        """
        配置 SQLite 测试连接。

        关闭驱动层隐式事务，并复用应用的外键配置。
        """

        dbapi_connection.isolation_level = None

        enable_sqlite_foreign_keys(
            dbapi_connection,
            connection_record,
        )

    @event.listens_for(engine, "begin")
    def begin_sqlite_transaction(
        connection,
    ) -> None:
        """
        显式开启SQLite事务。
        """

        connection.exec_driver_sql("BEGIN")

    try:
        yield engine

    finally:
        engine.dispose()

        if os.path.exists(test_db_path):
            os.remove(test_db_path)




@pytest.fixture
def db(test_engine):
    """
    创建测试数据库会话。

    每个测试结束后回滚事务，
    保证测试数据隔离。
    测试内部commit和rollback只影响Savepoint，
    测试结束统一回滚外层事务。
    """

    connection = test_engine.connect()

    transaction = connection.begin()

    SessionLocal = sessionmaker(
        autocommit=False,
        autoflush=False,
        bind=connection,
        join_transaction_mode="create_savepoint",
    )

    db = SessionLocal()

    try:
        yield db

    finally:
        db.close()
        if transaction.is_active:
            transaction.rollback()
        connection.close()


