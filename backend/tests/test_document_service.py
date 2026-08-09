from collections.abc import Sequence
from pathlib import Path

from fastapi import HTTPException
import pytest
from sqlalchemy import event

import app.api.knowledge as knowledge_api
from app.constants.document_status import DocumentStatus
from app.constants.processing_job_stage import ProcessingJobStage
from app.constants.processing_job_status import ProcessingJobStatus
from app.models.database.document import Document
from app.models.database.user import User
from app.models.database.processing_job import ProcessingJob
from app.repositories.document_chunk_repository import DocumentChunkRepository
from app.repositories.document_content_repository import DocumentContentRepository
from app.repositories.document_repository import DocumentRepository
from app.repositories.processing_job_repository import (
    ProcessingJobRepository,
)
from app.services.document_operation_policy import (
    DocumentOperationConflictError,
    DocumentOperationPolicy,
)
from app.services.document_service import DocumentService
from app.services.status_machine import StatusMachine
from app.services.storage_service import StorageService
from app.services.vector_store.base import VectorIndex, VectorIndexRecord

class FakeVectorIndex(VectorIndex):
    """模拟文档向量索引删除。"""

    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.deleted_document_ids: list[int] = []

    def ensure_collection(self) -> None:
        pass

    def upsert(self, records: Sequence[VectorIndexRecord]) -> None:
        pass

    def delete_by_document_id(self, document_id: int) -> None:
        self.deleted_document_ids.append(document_id)

        if self.error is not None:
            raise self.error

@pytest.fixture()
def vector_index() -> FakeVectorIndex:
    """创建测试向量索引。"""

    return FakeVectorIndex()

@pytest.fixture()
def document_service(tmp_path, vector_index: FakeVectorIndex) -> DocumentService:
    """
    创建 DocumentService。
    """

    processing_job_repository = ProcessingJobRepository()

    return DocumentService(
        storage_service=StorageService(
            storage_dir=str(tmp_path),
        ),
        document_repository=DocumentRepository(),
        document_content_repository=DocumentContentRepository(),
        document_chunk_repository=DocumentChunkRepository(),
        processing_job_repository=processing_job_repository,
        document_operation_policy=DocumentOperationPolicy(
            processing_job_repository=processing_job_repository,
        ),
        vector_index=vector_index,
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
    assert documents[0].active_job is None


def test_list_documents_returns_active_jobs_without_n_plus_one(
    db,
    document_service: DocumentService,
) -> None:
    """
    验证文档列表批量返回活动任务，
    并且列表读取固定执行两条SELECT语句。
    """

    pending_document = document_service.upload_document(
        db=db,
        filename="pending.txt",
        content=b"pending",
    )
    running_document = document_service.upload_document(
        db=db,
        filename="running.txt",
        content=b"running",
    )
    terminal_document = document_service.upload_document(
        db=db,
        filename="terminal.txt",
        content=b"terminal",
    )

    db.add_all([
        ProcessingJob(
            document_id=pending_document.id,
            job_type="full_pipeline",
            status=ProcessingJobStatus.PENDING.value,
            stage=ProcessingJobStage.QUEUED.value,
            progress=0,
        ),
        ProcessingJob(
            document_id=running_document.id,
            job_type="full_pipeline",
            status=ProcessingJobStatus.RUNNING.value,
            stage=ProcessingJobStage.EMBEDDING.value,
            progress=60,
        ),
        ProcessingJob(
            document_id=terminal_document.id,
            job_type="full_pipeline",
            status=ProcessingJobStatus.SUCCEEDED.value,
            stage=ProcessingJobStage.COMPLETED.value,
            progress=100,
        ),
    ])
    db.commit()
    db.expire_all()

    select_count = 0

    def count_select_statements(
        conn,
        cursor,
        statement,
        parameters,
        context,
        executemany,
    ) -> None:
        del conn, cursor, parameters, context, executemany

        nonlocal select_count

        if statement.lstrip().upper().startswith("SELECT"):
            select_count += 1

    engine = db.get_bind()

    event.listen(
        engine,
        "before_cursor_execute",
        count_select_statements,
    )

    try:
        documents = document_service.list_documents(
            db=db,
        )
    finally:
        event.remove(
            engine,
            "before_cursor_execute",
            count_select_statements,
        )

    documents_by_id = {
        document.id: document
        for document in documents
    }

    pending_item = documents_by_id[pending_document.id]
    running_item = documents_by_id[running_document.id]
    terminal_item = documents_by_id[terminal_document.id]

    assert pending_item.active_job is not None
    assert pending_item.active_job.status == ProcessingJobStatus.PENDING
    assert pending_item.active_job.stage == ProcessingJobStage.QUEUED
    assert pending_item.active_job.progress == 0

    assert running_item.active_job is not None
    assert running_item.active_job.status == ProcessingJobStatus.RUNNING
    assert running_item.active_job.stage == ProcessingJobStage.EMBEDDING
    assert running_item.active_job.progress == 60

    assert terminal_item.active_job is None
    assert select_count == 2


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
    vector_index: FakeVectorIndex,
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
        stage=ProcessingJobStage.COMPLETED.value,
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
    assert vector_index.deleted_document_ids == [document_id]
    assert not file_path.exists()
    assert deleted_document is None
    assert deleted_processing_job is None

def test_delete_document_keeps_local_data_when_vector_index_fails(
    db,
    tmp_path,
) -> None:
    """
    验证外部向量索引删除失败时，
    不删除本地文件和数据库记录。
    """

    vector_index = FakeVectorIndex(
        error=RuntimeError("qdrant unavailable")
    )

    processing_job_repository = ProcessingJobRepository()

    service = DocumentService(
        storage_service=StorageService(storage_dir=str(tmp_path)),
        document_repository=DocumentRepository(),
        document_content_repository=DocumentContentRepository(),
        document_chunk_repository=DocumentChunkRepository(),
        processing_job_repository=processing_job_repository,
        document_operation_policy=DocumentOperationPolicy(
            processing_job_repository=processing_job_repository,
        ),
        vector_index=vector_index,
    )

    document_info = service.upload_document(
        db=db,
        filename="vector-index-failed.txt",
        content=b"document content",
    )

    document = db.get(Document, document_info.id)

    assert document is not None

    document_id = document.id
    file_path = Path(document.path)

    with pytest.raises(
        RuntimeError,
        match="qdrant unavailable",
    ):
        service.delete_document(
            db=db,
            document_id=document_id,
        )

    db.expire_all()

    saved_document = db.get(Document, document_id)

    assert vector_index.deleted_document_ids == [document_id]
    assert saved_document is not None
    assert file_path.exists()

@pytest.mark.parametrize(
    "active_status",
    [
        ProcessingJobStatus.PENDING,
        ProcessingJobStatus.RUNNING,
    ],
)
def test_delete_document_rejects_active_processing_job_without_side_effects(
    db,
    document_service: DocumentService,
    vector_index: FakeVectorIndex,
    active_status: ProcessingJobStatus,
) -> None:
    """
    验证活动任务期间拒绝删除，
    且不会删除向量索引、本地文件或SQL记录。
    """

    document_info = document_service.upload_document(
        db=db,
        filename=f"active-{active_status.value}.txt",
        content=b"active processing job",
    )

    document = db.get(Document, document_info.id)

    assert document is not None

    document_id = document.id
    file_path = Path(document.path)

    job = ProcessingJob(
        document_id=document_id,
        job_type="full_pipeline",
        status=active_status.value,
        stage=(
            ProcessingJobStage.PARSING.value
            if active_status == ProcessingJobStatus.RUNNING
            else ProcessingJobStage.QUEUED.value
        ),
        progress=(
            10
            if active_status == ProcessingJobStatus.RUNNING
            else 0
        ),
    )

    db.add(job)
    db.commit()

    with pytest.raises(
        DocumentOperationConflictError,
        match=(
            "document has an active processing job "
            "and cannot be deleted"
        ),
    ):
        document_service.delete_document(
            db=db,
            document_id=document_id,
        )

    db.expire_all()

    assert db.get(Document, document_id) is not None
    assert db.get(ProcessingJob, job.id) is not None
    assert file_path.exists()
    assert vector_index.deleted_document_ids == []


@pytest.mark.parametrize(
    "terminal_status",
    [
        ProcessingJobStatus.SUCCEEDED,
        ProcessingJobStatus.FAILED,
    ],
)
def test_delete_document_allows_terminal_job_history(
    db,
    document_service: DocumentService,
    terminal_status: ProcessingJobStatus,
) -> None:
    """
    验证成功或失败的历史任务不会阻止删除。
    """

    document_info = document_service.upload_document(
        db=db,
        filename=f"terminal-{terminal_status.value}.txt",
        content=b"terminal processing job",
    )

    document = db.get(Document, document_info.id)

    assert document is not None

    job = ProcessingJob(
        document_id=document.id,
        job_type="full_pipeline",
        status=terminal_status.value,
        stage=(
            ProcessingJobStage.COMPLETED.value
            if terminal_status == ProcessingJobStatus.SUCCEEDED
            else ProcessingJobStage.INDEXING.value
        ),
        progress=(
            100
            if terminal_status == ProcessingJobStatus.SUCCEEDED
            else 85
        ),
    )

    db.add(job)
    db.commit()

    document_id = document.id

    document_service.delete_document(
        db=db,
        document_id=document_id,
    )

    db.expire_all()

    assert db.get(Document, document_id) is None


def test_delete_document_api_maps_operation_conflict_to_409(
    db,
    monkeypatch,
) -> None:
    """
    验证Router只负责把删除业务冲突转换为HTTP 409。
    """

    class ConflictDocumentService:
        def delete_document(
            self,
            db,
            document_id: int,
        ) -> None:
            raise DocumentOperationConflictError(
                "document has an active processing job "
                "and cannot be deleted"
            )

    monkeypatch.setattr(
        knowledge_api,
        "document_service",
        ConflictDocumentService(),
    )

    with pytest.raises(HTTPException) as exc_info:
        monkeypatch.setattr(
            knowledge_api,
            "_require_document_access",
            lambda db, document_id, current_user: None,
        )
        knowledge_api.delete_document(
            document_id=1,
            db=db,
            current_user=User(
                id=1,
                email="test@example.com",
                password_hash="hash",
                role="user",
                is_active=True,
            ),
        )

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail == (
        "document has an active processing job "
        "and cannot be deleted"
    )

