import pytest
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from sqlalchemy.orm import Session

from app.constants.document_status import DocumentStatus
from app.constants.document_status import DocumentStatus
from app.constants.processing_job_status import ProcessingJobStatus
from app.constants.processing_job_status import ProcessingJobStatus
from app.constants.processing_job_type import ProcessingJobType
from app.constants.processing_job_type import ProcessingJobType
from app.models.database.document import Document
from app.models.database.document import Document
from app.models.database.processing_job import ProcessingJob
from app.models.database.processing_job import ProcessingJob
from app.repositories.document_repository import DocumentRepository
from app.repositories.processing_job_repository import ProcessingJobRepository
from app.services.processing_job_service import (
    ActiveProcessingJobError,
    InvalidProcessingJobError,
    ProcessingJobService,
)
from app.services.status_machine import InvalidStatusTransitionError


def test_processing_job_defaults_and_persistence(
    db: Session,
) -> None:
    """
    验证任务默认状态和基础持久化。
    """

    document = Document(
        filename="processing-job-test.txt",
        stored_name="processing-job-test-stored.txt",
        path=(
            "tests/uploads/"
            "processing-job-test-stored.txt"
        ),
        size=100,
        status=DocumentStatus.UPLOADED.value,
    )

    db.add(document)
    db.flush()

    job = ProcessingJob(
        document_id=document.id,
        job_type=(
            ProcessingJobType
            .DOCUMENT_PROCESSING
            .value
        ),
    )

    db.add(job)
    db.commit()
    db.refresh(job)

    assert job.id is not None
    assert job.document_id == document.id

    assert (
        job.job_type
        == ProcessingJobType
        .DOCUMENT_PROCESSING
        .value
    )

    assert (
        job.status
        == ProcessingJobStatus.PENDING.value
    )

    assert job.progress == 0
    assert job.error_message is None
    assert job.started_at is None
    assert job.finished_at is None
    assert job.created_at is not None
    assert job.updated_at is not None

def build_processing_job_service(
) -> ProcessingJobService:
    """
    创建任务管理服务。
    """

    return ProcessingJobService(
        document_repository=DocumentRepository(),
        processing_job_repository=(
            ProcessingJobRepository()
        ),
    )


def create_document(
    db: Session,
    status: DocumentStatus,
    filename: str = "processing-job.txt",
) -> Document:
    """
    创建任务测试使用的文档。
    """

    document = Document(
        filename=filename,
        stored_name=f"stored-{filename}",
        path=f"tests/uploads/stored-{filename}",
        size=100,
        status=status.value,
    )

    db.add(document)
    db.commit()
    db.refresh(document)

    return document

def test_processing_job_service_completes_job(
    db: Session,
) -> None:
    """
    验证任务创建、运行、进度和成功流程。
    """

    document = create_document(
        db=db,
        status=DocumentStatus.UPLOADED,
    )

    service = build_processing_job_service()

    job = service.create_job(
        db=db,
        document_id=document.id,
        job_type=(
            ProcessingJobType
            .DOCUMENT_PROCESSING
        ),
    )

    assert (
        job.status
        == ProcessingJobStatus.PENDING.value
    )
    assert job.progress == 0

    started_job = service.start_job(
        db=db,
        job_id=job.id,
    )

    assert (
        started_job.status
        == ProcessingJobStatus.RUNNING.value
    )
    assert started_job.started_at is not None

    progressing_job = service.update_progress(
        db=db,
        job_id=job.id,
        progress=50,
    )

    assert progressing_job.progress == 50

    succeeded_job = service.succeed_job(
        db=db,
        job_id=job.id,
    )

    assert (
        succeeded_job.status
        == ProcessingJobStatus.SUCCEEDED.value
    )
    assert succeeded_job.progress == 100
    assert succeeded_job.finished_at is not None
    assert succeeded_job.error_message is None

def test_processing_job_service_rejects_duplicate_active_job(
    db: Session,
) -> None:
    """
    验证同一文档不能创建两个活动任务。
    """

    document = create_document(
        db=db,
        status=DocumentStatus.UPLOADED,
        filename="duplicate-job.txt",
    )

    service = build_processing_job_service()

    service.create_job(
        db=db,
        document_id=document.id,
        job_type=(
            ProcessingJobType
            .DOCUMENT_PROCESSING
        ),
    )

    with pytest.raises(
        ActiveProcessingJobError,
        match="already has an active job",
    ):
        service.create_job(
            db=db,
            document_id=document.id,
            job_type=(
                ProcessingJobType
                .DOCUMENT_PROCESSING
            ),
        )

def test_database_rejects_duplicate_active_jobs(
    db: Session,
) -> None:
    """
    验证数据库阻止同一文档存在两个活动任务。
    """

    document = create_document(
        db=db,
        status=DocumentStatus.UPLOADED,
        filename="database-constraint.txt",
    )

    first_job = ProcessingJob(
        document_id=document.id,
        job_type=(
            ProcessingJobType
            .DOCUMENT_PROCESSING
            .value
        ),
        status=(
            ProcessingJobStatus.PENDING.value
        ),
    )

    db.add(first_job)
    db.commit()

    second_job = ProcessingJob(
        document_id=document.id,
        job_type=(
            ProcessingJobType
            .EMBEDDING
            .value
        ),
        status=(
            ProcessingJobStatus.RUNNING.value
        ),
    )

    db.add(second_job)

    with pytest.raises(IntegrityError):
        db.commit()

    db.rollback()

def test_failed_job_allows_new_retry_job(
    db: Session,
) -> None:
    """
    验证失败任务保留历史，
    同时允许创建新的重试任务。
    """

    document = create_document(
        db=db,
        status=DocumentStatus.UPLOADED,
        filename="retry-job.txt",
    )

    service = build_processing_job_service()

    first_job = service.create_job(
        db=db,
        document_id=document.id,
        job_type=(
            ProcessingJobType
            .DOCUMENT_PROCESSING
        ),
    )

    service.start_job(
        db=db,
        job_id=first_job.id,
    )

    failed_job = service.fail_job(
        db=db,
        job_id=first_job.id,
        error_message="文档解析失败",
    )

    assert (
        failed_job.status
        == ProcessingJobStatus.FAILED.value
    )

    retry_job = service.create_job(
        db=db,
        document_id=document.id,
        job_type=(
            ProcessingJobType
            .DOCUMENT_PROCESSING
        ),
    )

    assert retry_job.id != failed_job.id

    assert (
        retry_job.status
        == ProcessingJobStatus.PENDING.value
    )

def test_failed_job_cannot_restart(
    db: Session,
) -> None:
    """
    验证failed是终态，
    重试不能复用旧任务。
    """

    document = create_document(
        db=db,
        status=DocumentStatus.UPLOADED,
        filename="terminal-job.txt",
    )

    service = build_processing_job_service()

    job = service.create_job(
        db=db,
        document_id=document.id,
        job_type=(
            ProcessingJobType
            .DOCUMENT_PROCESSING
        ),
    )

    service.start_job(
        db=db,
        job_id=job.id,
    )

    service.fail_job(
        db=db,
        job_id=job.id,
        error_message="测试失败",
    )

    with pytest.raises(
        InvalidStatusTransitionError,
        match="非法任务状态流转",
    ):
        service.start_job(
            db=db,
            job_id=job.id,
        )

def test_rejects_embedding_job_before_chunking(
    db: Session,
) -> None:
    """
    验证文档尚未切片时不能创建向量化任务。
    """

    document = create_document(
        db=db,
        status=DocumentStatus.UPLOADED,
        filename="invalid-embedding-job.txt",
    )

    service = build_processing_job_service()

    with pytest.raises(
        InvalidProcessingJobError,
        match="does not allow embedding job",
    ):
        service.create_job(
            db=db,
            document_id=document.id,
            job_type=(
                ProcessingJobType.EMBEDDING
            ),
        )

