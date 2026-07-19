import os
import subprocess

import pytest

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.config import get_settings


settings = get_settings()


@pytest.fixture(scope="session")
def test_engine():
    """
    创建测试数据库。

    使用 Alembic 初始化表结构。
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


    return engine



@pytest.fixture
def db(test_engine):

    SessionLocal = sessionmaker(
        autocommit=False,
        autoflush=False,
        bind=test_engine,
    )


    db = SessionLocal()

    try:
        yield db

    finally:
        db.close()


