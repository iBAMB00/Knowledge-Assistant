from pathlib import Path

import pytest

from app.constants.document_status import DocumentStatus
from app.constants.processing_job_status import ProcessingJobStatus
from app.models.database.document import Document
from app.models.database.processing_job import ProcessingJob
from app.repositories.document_chunk_repository import DocumentChunkRepository
from app.repositories.document_content_repository import DocumentContentRepository
from app.repositories.document_repository import DocumentRepository
from app.services.document_service import DocumentService
from app.services.status_machine import StatusMachine
from app.services.storage_service import StorageService

@pytest.fixture()
def document_service(tmp_path):
    """
    创建 DocumentService。
    """

    return DocumentService(
        storage_service=StorageService(
            storage_dir=str(tmp_path),
        ),
        document_repository=DocumentRepository(),
        document_content_repository=DocumentContentRepository(),
        document_chunk_repository=DocumentChunkRepository(),
    )


def test_upload_document(
    db,
    document_service,
):
    """
    测试上传文档。
    """

    document = document_service.upload_document(
        db=db,
        filename="test.txt",
        content=b"hello knowledge assistant",
    )

    db.commit()

    assert document.id is not None
    assert document.filename == "test.txt"
    assert document.size > 0


def test_list_documents(
    db,
    document_service,
):
    """
    测试查询文档列表。
    """

    document_service.upload_document(
        db=db,
        filename="test.txt",
        content=b"hello",
    )

    db.commit()

    documents = document_service.list_documents(
        db=db,
    )

    assert len(documents) == 1
    assert documents[0].filename == "test.txt"


def test_update_status(
    db,
    document_service,
):
    """
    测试文档状态更新。
    """

    document_info = document_service.upload_document(
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

    StatusMachine.transition_document(
        document=document,
        target_status=DocumentStatus.PARSING,
    )

    db.commit()

    assert document.status == (
        DocumentStatus.PARSING.value
    )

def test_delete_document(
    db,
    document_service,
):
    """
    测试删除文档，并级联清理处理任务。
    """

    document_info = document_service.upload_document(
        db=db,
        filename="test.txt",
        content=b"hello",
    )

    document = db.get(
        Document,
        document_info.id,
    )

    assert document is not None

    document_id = document.id
    file_path = Path(document.path)

    processing_job = ProcessingJob(
        document_id=document_id,
        job_type="full_pipeline",
        status=ProcessingJobStatus.SUCCEEDED.value,
        progress=100,
    )

    db.add(processing_job)
    db.commit()
    db.refresh(processing_job)

    processing_job_id = processing_job.id

    foreign_keys_enabled = (
        db.connection()
        .exec_driver_sql(
            "PRAGMA foreign_keys"
        )
        .scalar_one()
    )

    assert foreign_keys_enabled == 1
    assert file_path.exists()

    document_service.delete_document(
        db=db,
        document_id=document_id,
    )

    db.expire_all()

    deleted_document = db.get(
        Document,
        document_id,
    )

    deleted_processing_job = db.get(
        ProcessingJob,
        processing_job_id,
    )

    assert not file_path.exists()
    assert deleted_document is None
    assert deleted_processing_job is None