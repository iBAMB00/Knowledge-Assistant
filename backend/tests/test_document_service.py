from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.database import Base
from app.models.database.document import Document
from app.models.database.document_content import DocumentContent

from app.services.document_service import DocumentService
from app.services.storage_service import StorageService
from app.services.parser_service import ParserService
from app.services.document_processing_service import DocumentProcessingService

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
    创建 DocumentService。
    """

    return DocumentService(
        storage_service=StorageService(
            storage_dir=str(tmp_path),
        ),
        document_repository=DocumentRepository(),
        document_content_repository=DocumentContentRepository(),
    )


def test_upload_document(
    db,
    service,
):
    """
    测试上传文档。
    """

    document = service.upload_document(
        db=db,
        filename="test.txt",
        content=b"hello secure assistant",
    )

    db.commit()

    assert document.id is not None
    assert document.filename == "test.txt"
    assert document.size > 0


def test_list_documents(
    db,
    service,
):
    """
    测试查询文档列表。
    """

    service.upload_document(
        db=db,
        filename="test.txt",
        content=b"hello",
    )

    db.commit()

    documents = service.list_documents(
        db=db,
    )

    assert len(documents) == 1
    assert documents[0].filename == "test.txt"


def test_update_status(
    db,
    service,
):
    """
    测试文档状态更新。
    """

    document_info = service.upload_document(
        db=db,
        filename="test.txt",
        content=b"hello",
    )

    db.commit()

    document = (
        db.query(Document)
        .filter(
            Document.id == document_info.id
        )
        .first()
    )

    service.document_repository.update_status(
        db=db,
        document=document,
        status=DocumentStatus.PARSING.value,
    )

    db.commit()

    assert document.status == (
        DocumentStatus.PARSING.value
    )




def test_delete_document(
    db,
    service,
):
    """
    测试删除文档。
    """

    document_info = service.upload_document(
        db=db,
        filename="test.txt",
        content=b"hello",
    )

    db.commit()

    document = (
        db.query(Document)
        .filter(
            Document.id == document_info.id
        )
        .first()
    )

    file_path = Path(
        document.path
    )

    assert file_path.exists()

    service.delete_document(
        db=db,
        document_id=document.id,
    )

    db.commit()

    assert not file_path.exists()

    deleted_document = (
        db.query(Document)
        .filter(
            Document.id == document.id
        )
        .first()
    )

    assert deleted_document is None