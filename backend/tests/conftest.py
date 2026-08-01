import os
import subprocess

import pytest

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from app.core.config import get_settings


settings = get_settings()


@pytest.fixture(scope="session")
def test_engine():
    """
    创建测试数据库。

    使用Alembic初始化表结构，并显式控制
    SQLite事务，保证Savepoint正确参与外层事务。
    """

    test_db_path = "test.db"

    if os.path.exists(test_db_path):
        os.remove(test_db_path)


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
        关闭sqlite3驱动的隐式事务管理。

        后续BEGIN由SQLAlchemy显式发出，
        确保Savepoint属于外层事务。
        """

        dbapi_connection.isolation_level = None

        cursor = dbapi_connection.cursor()

        cursor.execute(
            "PRAGMA foreign_keys=ON"
        )

        cursor.close()

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


