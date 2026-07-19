from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.database import Base
from app.models.database.document import Document
from app.models.database.document_content import DocumentContent
from app.constants.document_status import DocumentStatus
from app.services.document_processing_service import DocumentProcessingService
from app.services.storage_service import StorageService
from app.services.parser_service import ParserService
from app.repositories.document_repository import DocumentRepository
from app.repositories.document_content_repository import DocumentContentRepository
from app.constants.document_status import DocumentStatus



# 测试独立数据库
engine = create_engine(
    "sqlite:///:memory:",
    connect_args={
        "check_same_thread": False,
    },
)

TestingSessionLocal = sessionmaker(
    bind=engine,
)


@pytest.fixture()
def db():
    """
    创建测试数据库。
    """

    Base.metadata.create_all(
        bind=engine,
    )

    session = TestingSessionLocal()

    try:
        yield session

    finally:
        session.close()

        Base.metadata.drop_all(
            bind=engine,
        )

@pytest.fixture()
def service(tmp_path):
    """
    创建文档处理服务。
    """

    return DocumentProcessingService(
        storage_service=StorageService(str(tmp_path)),
        document_repository=DocumentRepository(),
        document_content_repository=DocumentContentRepository(),
        parser_service=ParserService(),
    )


