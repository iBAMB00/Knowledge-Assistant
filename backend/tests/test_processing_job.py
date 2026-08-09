from collections.abc import Callable, Iterator
from datetime import datetime, timedelta

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.api.dependencies.auth import get_current_user
from app.api.knowledge import get_processing_job_dispatcher, router
from app.api.processing_job import router as processing_job_router
from app.constants.document_status import DocumentStatus
from app.constants.embedding_status import EmbeddingStatus
from app.constants.processing_job_stage import ProcessingJobStage
from app.constants.processing_job_status import ProcessingJobStatus
from app.constants.processing_job_type import ProcessingJobType
from app.core.database import get_db
from app.models.database.document import Document
from app.models.database.document_chunk import DocumentChunk
from app.models.database.document_content import DocumentContent
from app.models.database.knowledge_base import KnowledgeBase
from app.models.database.processing_job import ProcessingJob
from app.models.database.user import User
from app.repositories.document_chunk_repository import DocumentChunkRepository
from app.repositories.document_repository import DocumentRepository
from app.repositories.processing_job_repository import ProcessingJobRepository
from app.schemas.document_response import DocumentResponse
from app.services.processing_job_dispatcher import (
    ProcessingJobDispatchError,
    ProcessingJobDispatcher,
)
from app.services.processing_job_executor import ProcessingJobExecutor
from app.services.processing_job_recovery_service import ProcessingJobRecoveryService
from app.services.processing_job_retry_policy import ProcessingJobRetryPolicy
from app.services.processing_job_runner import ProcessingJobRunner
from app.services.processing_job_service import (
    ActiveProcessingJobError,
    InvalidProcessingJobError,
    ProcessingJobAlreadyClaimedError,
    ProcessingJobService,
)
from app.services.processing_job_service import ProcessingJobNotFoundError
from app.services.status_machine import InvalidStatusTransitionError
from app.tasks.processing_job import execute_processing_job, retry_policy

class FakeDocumentProcessingService:
    """模拟文档解析切片服务。"""

    def __init__(
        self,
        error: Exception | None = None,
        calls: list[str] | None = None,
    ) -> None:
        self.error = error
        self.calls = calls
        self.call_count = 0

    def process_document(
        self,
        db: Session,
        document_id: int,
        status_callback: (
            Callable[[DocumentStatus], None] | None
        ) = None,
    ) -> DocumentResponse:
        self.call_count += 1

        if self.calls is not None:
            self.calls.append("document_processing")

        if status_callback is not None:
            status_callback(DocumentStatus.PARSING)

        if self.error is not None:
            raise self.error

        if status_callback is not None:
            status_callback(DocumentStatus.CHUNKING)

        return DocumentResponse(
            id=document_id,
            knowledge_base_id=None,
            filename="executor-test.txt",
            stored_name="executor-test-stored.txt",
            size=100,
            status=DocumentStatus.CHUNKED.value,
            created_at=datetime.utcnow(),
        )

class FakeEmbeddingService:
    """模拟文档向量化服务。"""

    def __init__(
        self,
        processed_count: int = 2,
        error: Exception | None = None,
        calls: list[str] | None = None,
    ) -> None:
        self.processed_count = processed_count
        self.error = error
        self.calls = calls
        self.call_count = 0

    def process_document(
        self,
        db: Session,
        document_id: int,
        batch_size: int = 100,
    ) -> int:
        self.call_count += 1

        if self.calls is not None:
            self.calls.append("embedding")

        if self.error is not None:
            raise self.error

        return self.processed_count

class FakeRunnerSession:
    """
    Runner 单元测试使用的数据库 Session。
    """

    def __init__(self) -> None:
        self.rolled_back = False
        self.closed = False

    def rollback(self) -> None:
        self.rolled_back = True

    def close(self) -> None:
        self.closed = True


class FakeProcessingJobExecutor:
    """
    记录 Runner 传入的 Session 和任务 ID。
    """

    def __init__(self) -> None:
        self.received_db = None
        self.received_job_id: int | None = None

    def execute_job(self, db, job_id: int) -> None:
        self.received_db = db
        self.received_job_id = job_id


class FailingProcessingJobExecutor:
    """
    模拟执行任务时发生未捕获异常。
    """

    def execute_job(self, db, job_id: int) -> None:
        raise RuntimeError("unexpected executor error")

class FakeProcessingJobDispatcher:
    """API 测试使用的 Celery 派发器。"""

    def __init__(self) -> None:
        self.received_job_id: int | None = None
        self.error: Exception | None = None

    def dispatch(self, job_id: int) -> None:
        if self.error is not None:
            raise self.error
        self.received_job_id = job_id

class FakeVectorIndexService:
    """模拟外部向量索引同步服务。"""

    def __init__(
        self,
        indexed_count: int = 2,
        error: Exception | None = None,
        calls: list[str] | None = None,
    ) -> None:
        self.indexed_count = indexed_count
        self.error = error
        self.calls = calls
        self.call_count = 0

    def index_document(self, db: Session, document_id: int) -> int:
        self.call_count += 1

        if self.calls is not None:
            self.calls.append("vector_index")

        if self.error is not None:
            raise self.error

        return self.indexed_count

def get_or_create_processing_job_test_scope(
    db: Session,
) -> tuple[User, KnowledgeBase]:
    """创建 ProcessingJob API 测试共用的用户与知识库。"""

    owner = db.scalar(
        select(User).where(
            User.email == "processing-job-tests@example.com"
        )
    )

    if owner is None:
        owner = User(
            email="processing-job-tests@example.com",
            password_hash="test-password-hash",
            role="user",
            is_active=True,
        )
        db.add(owner)
        db.flush()

    knowledge_base = db.scalar(
        select(KnowledgeBase).where(
            KnowledgeBase.owner_id == owner.id,
            KnowledgeBase.name == "Processing Job Tests",
        )
    )

    if knowledge_base is None:
        knowledge_base = KnowledgeBase(
            owner_id=owner.id,
            name="Processing Job Tests",
            description="ProcessingJob API test scope",
        )
        db.add(knowledge_base)
        db.flush()

    return owner, knowledge_base

def build_processing_job_executor(
    document_processing_service: FakeDocumentProcessingService | None = None,
    embedding_service: FakeEmbeddingService | None = None,
    vector_index_service: FakeVectorIndexService | None = None,
) -> ProcessingJobExecutor:
    """创建任务执行器。"""

    document_repository = DocumentRepository()

    processing_job_service = ProcessingJobService(
        document_repository=document_repository,
        processing_job_repository=ProcessingJobRepository(),
    )

    return ProcessingJobExecutor(
        document_repository=document_repository,
        processing_job_service=processing_job_service,
        document_processing_service=(
            document_processing_service or FakeDocumentProcessingService()
        ),  # type: ignore[arg-type]
        embedding_service=embedding_service or FakeEmbeddingService(),  # type: ignore[arg-type]
        vector_index_service=vector_index_service,  # type: ignore[arg-type]
    )

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

    assert job.stage == ProcessingJobStage.QUEUED.value
    assert job.progress == 0
    assert job.error_message is None
    assert job.attempt_count == 0
    assert job.lease_expires_at is None
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
    _, knowledge_base = (
        get_or_create_processing_job_test_scope(db)
    )

    document = Document(
        knowledge_base_id=knowledge_base.id,
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
    assert job.stage == ProcessingJobStage.QUEUED.value
    assert job.progress == 0

    started_job = service.start_job(
        db=db,
        job_id=job.id,
    )

    assert (
        started_job.status
        == ProcessingJobStatus.RUNNING.value
    )
    assert started_job.stage == ProcessingJobStage.QUEUED.value
    assert started_job.started_at is not None

    parsing_job = service.update_stage(
        db=db,
        job_id=job.id,
        stage=ProcessingJobStage.PARSING,
        progress=10,
    )
    assert parsing_job.stage == ProcessingJobStage.PARSING.value
    assert parsing_job.progress == 10

    progressing_job = service.update_progress(
        db=db,
        job_id=job.id,
        progress=20,
    )
    assert progressing_job.progress == 20

    chunking_job = service.update_stage(
        db=db,
        job_id=job.id,
        stage=ProcessingJobStage.CHUNKING,
        progress=60,
    )
    assert chunking_job.stage == ProcessingJobStage.CHUNKING.value

    finalizing_job = service.update_stage(
        db=db,
        job_id=job.id,
        stage=ProcessingJobStage.FINALIZING,
        progress=95,
    )
    assert (
        finalizing_job.stage
        == ProcessingJobStage.FINALIZING.value
    )

    succeeded_job = service.succeed_job(
        db=db,
        job_id=job.id,
    )

    assert (
        succeeded_job.status
        == ProcessingJobStatus.SUCCEEDED.value
    )
    assert (
        succeeded_job.stage
        == ProcessingJobStage.COMPLETED.value
    )
    assert succeeded_job.progress == 100
    assert succeeded_job.finished_at is not None
    assert succeeded_job.error_message is None

def test_processing_job_service_rejects_progress_regression(
    db: Session,
) -> None:
    """验证任务进度不能倒退。"""

    document = create_document(
        db=db,
        status=DocumentStatus.CHUNKED,
        filename="progress-regression.txt",
    )
    service = build_processing_job_service()

    job = service.create_job(
        db=db,
        document_id=document.id,
        job_type=ProcessingJobType.EMBEDDING,
    )
    service.start_job(db=db, job_id=job.id)

    service.update_stage(
        db=db,
        job_id=job.id,
        stage=ProcessingJobStage.EMBEDDING,
        progress=60,
    )

    with pytest.raises(
        InvalidProcessingJobError,
        match="progress cannot decrease",
    ):
        service.update_progress(
            db=db,
            job_id=job.id,
            progress=50,
        )


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

def test_executor_completes_document_processing_job(
    db: Session,
) -> None:
    """
    验证执行器完成任务后保存succeeded。
    """

    document = create_document(
        db=db,
        status=DocumentStatus.UPLOADED,
        filename="executor-success.txt",
    )

    fake_service = (
        FakeDocumentProcessingService()
    )

    executor = build_processing_job_executor(
        document_processing_service=(
            fake_service
        ),
    )

    result = executor.process_document(
        db=db,
        document_id=document.id,
    )

    assert (
        result.status
        == DocumentStatus.CHUNKED.value
    )

    assert fake_service.call_count == 1

    saved_job = (
        ProcessingJobRepository()
        .find_latest_by_document_id(
            db=db,
            document_id=document.id,
        )
    )

    assert saved_job is not None

    assert (
        saved_job.status
        == ProcessingJobStatus.SUCCEEDED.value
    )

    assert (
        saved_job.stage
        == ProcessingJobStage.COMPLETED.value
    )
    assert saved_job.progress == 100
    assert saved_job.started_at is not None
    assert saved_job.finished_at is not None

def test_executor_marks_job_failed_without_leaking_error(
    db: Session,
) -> None:
    """
    验证业务失败后任务进入failed，
    且不保存底层异常详情。
    """

    document = create_document(
        db=db,
        status=DocumentStatus.UPLOADED,
        filename="executor-failed.txt",
    )

    fake_service = (
        FakeDocumentProcessingService(
            error=RuntimeError(
                "secret-path=/private/document.txt"
            )
        )
    )

    executor = build_processing_job_executor(
        document_processing_service=(
            fake_service
        ),
    )

    with pytest.raises(
        RuntimeError,
        match="secret-path",
    ):
        executor.process_document(
            db=db,
            document_id=document.id,
        )

    saved_job = (
        ProcessingJobRepository()
        .find_latest_by_document_id(
            db=db,
            document_id=document.id,
        )
    )

    assert saved_job is not None

    assert (
        saved_job.status
        == ProcessingJobStatus.FAILED.value
    )

    assert saved_job.progress == 10
    
    assert saved_job.error_message == (
        "文档解析或切片失败"
    )

    assert (
        "secret-path"
        not in saved_job.error_message
    )

def test_executor_does_not_create_job_for_completed_embedding(
    db: Session,
) -> None:
    """
    验证已完成文档重复向量化时保持幂等，
    不创建没有实际工作的任务。
    """

    document = create_document(
        db=db,
        status=DocumentStatus.COMPLETED,
        filename="executor-noop.txt",
    )

    fake_embedding_service = (
        FakeEmbeddingService(
            processed_count=0,
        )
    )

    executor = build_processing_job_executor(
        embedding_service=(
            fake_embedding_service
        ),
    )

    processed_count = executor.embed_document(
        db=db,
        document_id=document.id,
    )

    assert processed_count == 0
    assert fake_embedding_service.call_count == 1

    saved_job = (
        ProcessingJobRepository()
        .find_latest_by_document_id(
            db=db,
            document_id=document.id,
        )
    )

    assert saved_job is None

def test_get_latest_document_job(
    db: Session,
) -> None:
    """
    验证可以查询文档最近一次任务。
    """

    document = create_document(
        db=db,
        status=DocumentStatus.UPLOADED,
        filename="latest-job.txt",
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

    service.fail_job(
        db=db,
        job_id=first_job.id,
        error_message="处理失败",
    )

    second_job = service.create_job(
        db=db,
        document_id=document.id,
        job_type=(
            ProcessingJobType
            .DOCUMENT_PROCESSING
        ),
    )

    latest_job = (
        service.get_latest_document_job(
            db=db,
            document_id=document.id,
        )
    )

    assert latest_job.id == second_job.id

    assert (
        latest_job.status
        == ProcessingJobStatus.PENDING.value
    )

def test_get_latest_document_job_rejects_missing_job(
    db: Session,
) -> None:
    """
    验证文档没有处理任务时抛出异常。
    """

    document = create_document(
        db=db,
        status=DocumentStatus.UPLOADED,
        filename="no-job.txt",
    )

    service = build_processing_job_service()

    with pytest.raises(
        ProcessingJobNotFoundError,
        match="processing job not found",
    ):
        service.get_latest_document_job(
            db=db,
            document_id=document.id,
        )

def test_get_latest_document_job_rejects_missing_document(
    db: Session,
) -> None:
    """
    验证文档不存在时抛出异常。
    """

    service = build_processing_job_service()

    with pytest.raises(
        ValueError,
        match="document not found",
    ):
        service.get_latest_document_job(
            db=db,
            document_id=999999,
        )


def test_processing_job_runner_uses_independent_session() -> None:
    """
    验证 Runner 创建独立 Session 并调用 Executor。
    """

    fake_session = FakeRunnerSession()
    fake_executor = FakeProcessingJobExecutor()

    runner = ProcessingJobRunner(
        session_factory=lambda: fake_session,
        executor=fake_executor,
    )

    runner.run(job_id=10)

    assert fake_executor.received_db is fake_session
    assert fake_executor.received_job_id == 10
    assert fake_session.rolled_back is False
    assert fake_session.closed is True


def test_processing_job_runner_rolls_back_on_error() -> None:
    """
    验证 Executor 抛出异常时 Runner 回滚并关闭 Session。
    """

    fake_session = FakeRunnerSession()

    runner = ProcessingJobRunner(
        session_factory=lambda: fake_session,
        executor=FailingProcessingJobExecutor(),
    )

    with pytest.raises(RuntimeError, match="unexpected executor error"):
        runner.run(job_id=11)

    assert fake_session.rolled_back is True
    assert fake_session.closed is True

class FakeCeleryTask:
    """记录 Dispatcher 提交给 Celery 的 job_id。"""

    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.received_job_id: int | None = None

    def delay(self, job_id: int) -> None:
        if self.error is not None:
            raise self.error
        self.received_job_id = job_id


class FakeWorkerRunner:
    """记录 Celery 业务任务交给 Worker Runner 的 job_id。"""

    def __init__(self) -> None:
        self.received_job_id: int | None = None
        self.released_job_id: int | None = None
        self.failed_job_id: int | None = None

    def run_worker(self, job_id: int, force_resume: bool = False) -> bool:
        self.received_job_id = job_id
        return True

    def release_for_retry(self, job_id: int) -> None:
        self.released_job_id = job_id

    def fail(self, job_id: int) -> None:
        self.failed_job_id = job_id


def test_processing_job_dispatcher_sends_only_job_id() -> None:
    """验证派发器只通过 Celery Task 发送持久化任务 ID。"""
    fake_task = FakeCeleryTask()
    dispatcher = ProcessingJobDispatcher(task=fake_task)

    dispatcher.dispatch(job_id=21)

    assert fake_task.received_job_id == 21


def test_processing_job_dispatcher_wraps_broker_error() -> None:
    """验证 Broker/Celery 派发异常被转换为明确业务异常。"""
    dispatcher = ProcessingJobDispatcher(
        task=FakeCeleryTask(error=RuntimeError("redis unavailable"))
    )

    with pytest.raises(ProcessingJobDispatchError):
        dispatcher.dispatch(job_id=22)


def test_celery_processing_job_task_calls_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证 Worker 任务根据 job_id 调用独立 Session Runner。"""
    fake_runner = FakeWorkerRunner()
    monkeypatch.setattr(
        "app.tasks.processing_job.get_processing_job_runner",
        lambda: fake_runner,
    )

    execute_processing_job.run(23)

    assert fake_runner.received_job_id == 23


def test_latest_processing_job_route_is_not_duplicated() -> None:
    """
    验证最新任务查询路径只声明一次。

    直接检查业务Router，不依赖FastAPI内部
    对include_router的组织方式。
    """

    target_path = (
        "/documents/{document_id}"
        "/processing-jobs/latest"
    )

    business_routes = [
        *router.routes,
        *processing_job_router.routes,
    ]

    matched_routes = [
        route
        for route in business_routes
        if (
            getattr(route, "path", "").rstrip("/") == target_path
            and "GET" in (getattr(route, "methods", set()) or set())
        )
    ]

    registered_routes = [
        {
            "path": getattr(route, "path", None),
            "methods": sorted(
                getattr(route, "methods", set()) or set()
            ),
            "name": getattr(route, "name", None),
        }
        for route in business_routes
    ]

    assert len(matched_routes) == 1, (
        "最新任务查询路由声明数量错误，"
        f"matched={len(matched_routes)}, "
        f"routes={registered_routes}"
    )

@pytest.fixture
def processing_job_client(
    db: Session,
) -> Iterator[tuple[TestClient, FakeProcessingJobDispatcher]]:
    """创建后台任务接口测试客户端。"""

    app = FastAPI()
    app.include_router(router)
    app.include_router(processing_job_router)

    fake_dispatcher = FakeProcessingJobDispatcher()

    current_user, _ = (
        get_or_create_processing_job_test_scope(db)
    )

    db.commit()

    def override_get_db():
        yield db

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_current_user] = lambda: current_user
    app.dependency_overrides[get_processing_job_dispatcher] = lambda: fake_dispatcher

    with TestClient(app) as client:
        yield client, fake_dispatcher

def test_create_document_processing_job_api(
    db: Session,
    processing_job_client: tuple[TestClient, FakeProcessingJobDispatcher],
) -> None:
    """验证接口创建任务并交给后台Runner。"""

    client, fake_dispatcher = processing_job_client

    document = create_document(
        db=db,
        filename="async-processing.txt",
        status=DocumentStatus.UPLOADED,
    )

    response = client.post(
        f"/documents/{document.id}/processing-jobs",
        json={
            "job_type": "document_processing",
        },
    )

    assert response.status_code == 202

    body = response.json()

    assert body["document_id"] == document.id
    assert body["job_type"] == "document_processing"
    assert body["status"] == "pending"
    assert body["stage"] == "queued"
    assert body["progress"] == 0
    assert fake_dispatcher.received_job_id == body["id"]

def test_create_processing_job_marks_failed_when_dispatch_fails(
    db: Session,
    processing_job_client: tuple[TestClient, FakeProcessingJobDispatcher],
) -> None:
    """验证 Broker 派发失败不会遗留永久 pending 活动任务。"""
    client, fake_dispatcher = processing_job_client
    fake_dispatcher.error = ProcessingJobDispatchError(
        "processing job dispatch failed"
    )

    document = create_document(
        db=db,
        filename="dispatch-failed.txt",
        status=DocumentStatus.UPLOADED,
    )

    response = client.post(
        f"/documents/{document.id}/processing-jobs",
        json={"job_type": "full_pipeline"},
    )

    assert response.status_code == 503
    assert response.json() == {"detail": "处理任务派发失败"}

    failed_job = ProcessingJobRepository().find_latest_by_document_id(
        db=db,
        document_id=document.id,
    )
    assert failed_job is not None
    assert failed_job.status == ProcessingJobStatus.FAILED.value
    assert failed_job.stage == ProcessingJobStage.QUEUED.value
    assert failed_job.progress == 0
    assert failed_job.error_message == "处理任务派发失败"

    fake_dispatcher.error = None
    retry_response = client.post(
        f"/documents/{document.id}/processing-jobs",
        json={"job_type": "full_pipeline"},
    )
    assert retry_response.status_code == 202
    assert retry_response.json()["id"] != failed_job.id


def test_create_processing_job_returns_404(
    processing_job_client: tuple[TestClient, FakeProcessingJobDispatcher],
) -> None:
    """验证文档不存在时不提交后台任务。"""

    client, fake_dispatcher = processing_job_client

    response = client.post(
        "/documents/999999/processing-jobs",
        json={
            "job_type": "document_processing",
        },
    )

    assert response.status_code == 404
    assert response.json() == {
        "detail": "document not found",
    }
    assert fake_dispatcher.received_job_id is None

def test_executor_completes_full_pipeline_job(
    db: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    验证完整流水线按顺序执行已有任务，
    保存阶段进度并最终标记成功。
    """

    document = create_document(
        db=db,
        status=DocumentStatus.UPLOADED,
        filename="full-pipeline.txt",
    )

    calls: list[str] = []
    stage_updates: list[
        tuple[ProcessingJobStage, int]
    ] = []

    vector_index_service = FakeVectorIndexService(calls=calls)

    document_processing_service = (
        FakeDocumentProcessingService(
            calls=calls,
        )
    )

    embedding_service = FakeEmbeddingService(
        calls=calls,
    )

    executor = build_processing_job_executor(
        document_processing_service=document_processing_service,
        embedding_service=embedding_service,
        vector_index_service=vector_index_service,
    )

    original_update_stage = (
        executor
        .processing_job_service
        .update_stage
    )

    def record_stage(
        db: Session,
        job_id: int,
        stage: ProcessingJobStage,
        progress: int,
    ) -> ProcessingJob:
        """记录执行器提交的任务阶段和进度。"""

        stage_updates.append((stage, progress))

        return original_update_stage(
            db=db,
            job_id=job_id,
            stage=stage,
            progress=progress,
        )

    monkeypatch.setattr(
        executor.processing_job_service,
        "update_stage",
        record_stage,
    )

    job = (
        executor.processing_job_service
        .create_job(
            db=db,
            document_id=document.id,
            job_type=(
                ProcessingJobType.FULL_PIPELINE
            ),
        )
    )

    result = executor.execute_job(
        db=db,
        job_id=job.id,
    )

    saved_job = (
        executor.processing_job_service
        .get_job(
            db=db,
            job_id=job.id,
        )
    )

    assert result == 2

    assert calls == [
        "document_processing",
        "embedding",
        "vector_index",
    ]
    
    assert vector_index_service.call_count == 1

    assert stage_updates == [
        (ProcessingJobStage.PARSING, 10),
        (ProcessingJobStage.CHUNKING, 35),
        (ProcessingJobStage.EMBEDDING, 60),
        (ProcessingJobStage.INDEXING, 85),
        (ProcessingJobStage.FINALIZING, 95),
    ]

    assert (
        document_processing_service.call_count
        == 1
    )

    assert embedding_service.call_count == 1

    assert saved_job.status == (
        ProcessingJobStatus.SUCCEEDED.value
    )

    assert (
        saved_job.stage
        == ProcessingJobStage.COMPLETED.value
    )
    assert saved_job.progress == 100
    assert saved_job.error_message is None
    assert saved_job.started_at is not None
    assert saved_job.finished_at is not None

def test_full_pipeline_keeps_progress_when_embedding_fails(
    db: Session,
) -> None:
    """
    验证完整流水线向量化失败时，
    保留解析切片已完成的进度。
    """

    document = create_document(
        db=db,
        status=DocumentStatus.UPLOADED,
        filename=(
            "full-pipeline-embedding-failed.txt"
        ),
    )

    calls: list[str] = []

    document_processing_service = (
        FakeDocumentProcessingService(
            calls=calls,
        )
    )

    embedding_service = FakeEmbeddingService(
        calls=calls,
        error=RuntimeError(
            "embedding service unavailable"
        ),
    )

    executor = build_processing_job_executor(
        document_processing_service=(
            document_processing_service
        ),
        embedding_service=embedding_service,
    )

    job = (
        executor.processing_job_service
        .create_job(
            db=db,
            document_id=document.id,
            job_type=(
                ProcessingJobType.FULL_PIPELINE
            ),
        )
    )

    with pytest.raises(
        RuntimeError,
        match="embedding service unavailable",
    ):
        executor.execute_job(
            db=db,
            job_id=job.id,
        )

    saved_job = (
        executor.processing_job_service
        .get_job(
            db=db,
            job_id=job.id,
        )
    )

    assert calls == [
        "document_processing",
        "embedding",
    ]

    assert (
        document_processing_service.call_count
        == 1
    )

    assert embedding_service.call_count == 1

    assert saved_job.status == (
        ProcessingJobStatus.FAILED.value
    )

    assert (
        saved_job.stage
        == ProcessingJobStage.EMBEDDING.value
    )
    assert saved_job.progress == 60

    assert saved_job.error_message == (
        "文档完整处理失败"
    )

    assert saved_job.started_at is not None
    assert saved_job.finished_at is not None




def test_get_latest_document_processing_job_api(
    db: Session,
    processing_job_client: tuple[
        TestClient,
        FakeProcessingJobDispatcher,
    ],
) -> None:
    """验证接口返回文档最近一次处理任务。"""

    client, _ = processing_job_client

    document = create_document(
        db=db,
        status=DocumentStatus.UPLOADED,
        filename="latest-job-api.txt",
    )

    service = build_processing_job_service()

    job = service.create_job(
        db=db,
        document_id=document.id,
        job_type=ProcessingJobType.FULL_PIPELINE,
    )

    response = client.get(
        f"/documents/{document.id}/processing-jobs/latest"
    )

    assert response.status_code == 200

    body = response.json()

    assert body["id"] == job.id
    assert body["document_id"] == document.id
    assert body["job_type"] == "full_pipeline"
    assert body["status"] == "pending"
    assert body["stage"] == "queued"
    assert body["progress"] == 0
    assert body["error_message"] is None
    assert body["created_at"] is not None
    assert body["updated_at"] is not None
    assert body["started_at"] is None
    assert body["finished_at"] is None


def test_get_latest_processing_job_returns_404_when_job_missing(
    db: Session,
    processing_job_client: tuple[
        TestClient,
        FakeProcessingJobDispatcher,
    ],
) -> None:
    """验证文档没有处理任务时接口返回404。"""

    client, _ = processing_job_client

    document = create_document(
        db=db,
        status=DocumentStatus.UPLOADED,
        filename="no-latest-job-api.txt",
    )

    response = client.get(
        f"/documents/{document.id}/processing-jobs/latest"
    )

    assert response.status_code == 404
    assert response.json() == {
        "detail": "processing job not found",
    }


def test_get_latest_processing_job_returns_404_when_document_missing(
    processing_job_client: tuple[
        TestClient,
        FakeProcessingJobDispatcher,
    ],
) -> None:
    """验证文档不存在时接口返回404。"""

    client, _ = processing_job_client

    response = client.get(
        "/documents/999999/processing-jobs/latest"
    )

    assert response.status_code == 404
    assert response.json() == {
        "detail": "document not found",
    }

def test_create_processing_job_returns_409_when_active_job_exists(
    db: Session,
    processing_job_client: tuple[
        TestClient,
        FakeProcessingJobDispatcher,
    ],
) -> None:
    """
    验证同一文档已有活动任务时，
    接口拒绝创建第二个任务。
    """

    client, fake_dispatcher = processing_job_client

    document = create_document(
        db=db,
        status=DocumentStatus.UPLOADED,
        filename="duplicate-active-job-api.txt",
    )

    first_response = client.post(
        f"/documents/{document.id}/processing-jobs",
        json={
            "job_type": "full_pipeline",
        },
    )

    assert first_response.status_code == 202

    first_body = first_response.json()

    second_response = client.post(
        f"/documents/{document.id}/processing-jobs",
        json={
            "job_type": "full_pipeline",
        },
    )

    assert second_response.status_code == 409

    assert second_response.json() == {
        "detail": (
            "document already has an active job"
        ),
    }

    assert (
        fake_dispatcher.received_job_id
        == first_body["id"]
    )

    active_job = (
        ProcessingJobRepository()
        .find_active_by_document_id(
            db=db,
            document_id=document.id,
        )
    )

    assert active_job is not None
    assert active_job.id == first_body["id"]

    assert active_job.status == (
        ProcessingJobStatus.PENDING.value
    )

def test_create_processing_job_allows_retry_after_failure(
    db: Session,
    processing_job_client: tuple[
        TestClient,
        FakeProcessingJobDispatcher,
    ],
) -> None:
    """
    验证旧任务失败后保留历史，
    同时允许接口创建新的重试任务。
    """

    client, fake_dispatcher = processing_job_client

    document = create_document(
        db=db,
        status=DocumentStatus.UPLOADED,
        filename="retry-failed-job-api.txt",
    )

    service = build_processing_job_service()

    failed_job = service.create_job(
        db=db,
        document_id=document.id,
        job_type=ProcessingJobType.FULL_PIPELINE,
    )

    service.start_job(
        db=db,
        job_id=failed_job.id,
    )

    service.fail_job(
        db=db,
        job_id=failed_job.id,
        error_message="文档完整处理失败",
    )

    response = client.post(
        f"/documents/{document.id}/processing-jobs",
        json={
            "job_type": "full_pipeline",
        },
    )

    assert response.status_code == 202

    body = response.json()

    assert body["id"] != failed_job.id
    assert body["document_id"] == document.id
    assert body["job_type"] == "full_pipeline"
    assert body["status"] == "pending"
    assert body["stage"] == "queued"
    assert body["progress"] == 0
    assert body["error_message"] is None

    assert fake_dispatcher.received_job_id == body["id"]

    saved_failed_job = service.get_job(
        db=db,
        job_id=failed_job.id,
    )

    assert saved_failed_job.status == (
        ProcessingJobStatus.FAILED.value
    )

    assert saved_failed_job.error_message == (
        "文档完整处理失败"
    )

    latest_job = service.get_latest_document_job(
        db=db,
        document_id=document.id,
    )

    assert latest_job.id == body["id"]

    assert latest_job.status == (
        ProcessingJobStatus.PENDING.value
    )

def test_list_document_processing_jobs_api(
    db: Session,
    processing_job_client: tuple[
        TestClient,
        FakeProcessingJobDispatcher,
    ],
) -> None:
    """
    验证接口按最新任务优先返回文档任务历史。
    """

    client, _ = processing_job_client

    document = create_document(
        db=db,
        status=DocumentStatus.UPLOADED,
        filename="processing-job-history.txt",
    )

    service = build_processing_job_service()

    first_job = service.create_job(
        db=db,
        document_id=document.id,
        job_type=(
            ProcessingJobType.DOCUMENT_PROCESSING
        ),
    )

    service.start_job(
        db=db,
        job_id=first_job.id,
    )

    service.fail_job(
        db=db,
        job_id=first_job.id,
        error_message="文档解析或切片失败",
    )

    second_job = service.create_job(
        db=db,
        document_id=document.id,
        job_type=ProcessingJobType.FULL_PIPELINE,
    )

    response = client.get(
        f"/documents/{document.id}/processing-jobs"
    )

    assert response.status_code == 200

    body = response.json()

    assert len(body) == 2

    assert [
        item["id"]
        for item in body
    ] == [
        second_job.id,
        first_job.id,
    ]

    assert body[0]["document_id"] == document.id
    assert body[0]["job_type"] == "full_pipeline"
    assert body[0]["status"] == "pending"
    assert body[0]["stage"] == "queued"
    assert body[0]["progress"] == 0
    assert body[0]["error_message"] is None

    assert body[1]["document_id"] == document.id

    assert (
        body[1]["job_type"]
        == "document_processing"
    )

    assert body[1]["status"] == "failed"
    assert body[1]["stage"] == "queued"

    assert body[1]["error_message"] == (
        "文档解析或切片失败"
    )

def test_list_document_processing_jobs_returns_empty_list(
    db: Session,
    processing_job_client: tuple[
        TestClient,
        FakeProcessingJobDispatcher,
    ],
) -> None:
    """
    验证文档存在但没有任务时返回空列表。
    """

    client, _ = processing_job_client

    document = create_document(
        db=db,
        status=DocumentStatus.UPLOADED,
        filename="empty-processing-job-history.txt",
    )

    response = client.get(
        f"/documents/{document.id}/processing-jobs"
    )

    assert response.status_code == 200
    assert response.json() == []

def test_list_document_processing_jobs_returns_404(
    processing_job_client: tuple[
        TestClient,
        FakeProcessingJobDispatcher,
    ],
) -> None:
    """
    验证文档不存在时任务列表接口返回404。
    """

    client, _ = processing_job_client

    response = client.get(
        "/documents/999999/processing-jobs"
    )

    assert response.status_code == 404

    assert response.json() == {
        "detail": "document not found",
    }

def test_executor_reindexes_completed_document_with_job(db: Session) -> None:
    """验证completed文档启用外部索引时创建可追踪任务。"""

    document = create_document(
        db=db,
        status=DocumentStatus.COMPLETED,
        filename="completed-reindex.txt",
    )

    calls: list[str] = []
    embedding_service = FakeEmbeddingService(processed_count=0, calls=calls)
    vector_index_service = FakeVectorIndexService(calls=calls)

    executor = build_processing_job_executor(
        embedding_service=embedding_service,
        vector_index_service=vector_index_service,
    )

    processed_count = executor.embed_document(db=db, document_id=document.id)

    saved_job = ProcessingJobRepository().find_latest_by_document_id(
        db=db, document_id=document.id
    )

    assert processed_count == 0
    assert calls == ["embedding", "vector_index"]
    assert embedding_service.call_count == 1
    assert vector_index_service.call_count == 1
    assert saved_job is not None
    assert saved_job.status == ProcessingJobStatus.SUCCEEDED.value
    assert saved_job.stage == ProcessingJobStage.COMPLETED.value
    assert saved_job.progress == 100

def test_full_pipeline_marks_job_failed_when_vector_index_fails(
    db: Session,
) -> None:
    """验证外部向量索引失败后任务进入failed。"""

    document = create_document(
        db=db,
        status=DocumentStatus.UPLOADED,
        filename="vector-index-failed.txt",
    )

    calls: list[str] = []

    executor = build_processing_job_executor(
        document_processing_service=FakeDocumentProcessingService(calls=calls),
        embedding_service=FakeEmbeddingService(calls=calls),
        vector_index_service=FakeVectorIndexService(
            calls=calls,
            error=RuntimeError("qdrant unavailable"),
        ),
    )

    job = executor.processing_job_service.create_job(
        db=db,
        document_id=document.id,
        job_type=ProcessingJobType.FULL_PIPELINE,
    )

    with pytest.raises(RuntimeError, match="qdrant unavailable"):
        executor.execute_job(db=db, job_id=job.id)

    saved_job = executor.processing_job_service.get_job(db=db, job_id=job.id)

    assert calls == [
        "document_processing",
        "embedding",
        "vector_index",
    ]
    assert saved_job.status == ProcessingJobStatus.FAILED.value
    assert saved_job.stage == ProcessingJobStage.INDEXING.value
    assert saved_job.progress == 85
    assert saved_job.error_message == "文档完整处理失败"




def test_worker_claim_uses_lease_and_allows_resume_after_release(db: Session) -> None:
    """验证 Worker 租约阻止并发领取，并允许瞬时失败后重新领取。"""
    document = create_document(
        db=db,
        status=DocumentStatus.UPLOADED,
        filename="worker-lease.txt",
    )
    service = ProcessingJobService(
        document_repository=DocumentRepository(),
        processing_job_repository=ProcessingJobRepository(),
        lease_seconds=60,
    )
    job = service.create_job(
        db=db,
        document_id=document.id,
        job_type=ProcessingJobType.FULL_PIPELINE,
    )

    first_claim = service.claim_job_for_worker(db=db, job_id=job.id)
    assert first_claim is not None
    assert first_claim.resumed is False

    running_job = service.get_job(db=db, job_id=job.id)
    assert running_job.status == ProcessingJobStatus.RUNNING.value
    assert running_job.attempt_count == 1
    assert running_job.lease_expires_at is not None

    with pytest.raises(ProcessingJobAlreadyClaimedError):
        service.claim_job_for_worker(db=db, job_id=job.id)

    service.release_job_for_retry(db=db, job_id=job.id)
    second_claim = service.claim_job_for_worker(db=db, job_id=job.id)
    assert second_claim is not None
    assert second_claim.resumed is True
    assert service.get_job(db=db, job_id=job.id).attempt_count == 2


def test_worker_claim_reclaims_expired_running_job(db: Session) -> None:
    """验证 Worker 异常退出后，过期租约可被新 Worker 接管。"""
    document = create_document(
        db=db,
        status=DocumentStatus.UPLOADED,
        filename="expired-lease.txt",
    )
    service = ProcessingJobService(
        document_repository=DocumentRepository(),
        processing_job_repository=ProcessingJobRepository(),
        lease_seconds=60,
    )
    job = service.create_job(
        db=db,
        document_id=document.id,
        job_type=ProcessingJobType.FULL_PIPELINE,
    )
    service.claim_job_for_worker(db=db, job_id=job.id)

    saved_job = service.get_job(db=db, job_id=job.id)
    saved_job.lease_expires_at = datetime.utcnow() - timedelta(seconds=1)
    db.commit()

    claim = service.claim_job_for_worker(db=db, job_id=job.id)
    assert claim is not None
    assert claim.resumed is True
    assert service.get_job(db=db, job_id=job.id).attempt_count == 2


def test_worker_claim_skips_terminal_duplicate(db: Session) -> None:
    """验证已成功/失败的业务任务收到重复 Celery 消息时不再执行。"""
    document = create_document(
        db=db,
        status=DocumentStatus.CHUNKED,
        filename="terminal-duplicate.txt",
    )
    service = build_processing_job_service()
    job = service.create_job(
        db=db,
        document_id=document.id,
        job_type=ProcessingJobType.EMBEDDING,
    )
    service.start_job(db=db, job_id=job.id)
    service.update_stage(
        db=db,
        job_id=job.id,
        stage=ProcessingJobStage.EMBEDDING,
        progress=10,
    )
    service.update_stage(
        db=db,
        job_id=job.id,
        stage=ProcessingJobStage.FINALIZING,
        progress=95,
    )
    service.succeed_job(db=db, job_id=job.id)

    assert service.claim_job_for_worker(db=db, job_id=job.id) is None


def test_recovery_resets_interrupted_embedding_state(db: Session) -> None:
    """验证 Worker 崩溃遗留的 processing Chunk 只回退为可重试状态。"""
    document = create_document(
        db=db,
        status=DocumentStatus.EMBEDDING,
        filename="recover-embedding.txt",
    )
    content = DocumentContent(
        document_id=document.id,
        content="用于恢复测试的正文",
        parser_type="test",
        parser_version="1",
    )
    db.add(content)
    db.flush()
    processing_chunk = DocumentChunk(
        document_content_id=content.id,
        chunk_index=0,
        content="processing",
        chunk_strategy="recursive_character",
        embedding_status=EmbeddingStatus.PROCESSING.value,
    )
    completed_chunk = DocumentChunk(
        document_content_id=content.id,
        chunk_index=1,
        content="completed",
        chunk_strategy="recursive_character",
        embedding_status=EmbeddingStatus.COMPLETED.value,
    )
    db.add_all([processing_chunk, completed_chunk])
    db.commit()

    job = ProcessingJob(
        document_id=document.id,
        job_type=ProcessingJobType.FULL_PIPELINE.value,
        status=ProcessingJobStatus.RUNNING.value,
        stage=ProcessingJobStage.EMBEDDING.value,
        progress=60,
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    recovery = ProcessingJobRecoveryService(
        document_repository=DocumentRepository(),
        document_chunk_repository=DocumentChunkRepository(),
    )
    recovery.prepare_for_resume(db=db, job=job)

    db.refresh(document)
    db.refresh(processing_chunk)
    db.refresh(completed_chunk)
    assert document.status == DocumentStatus.EMBEDDING_FAILED.value
    assert processing_chunk.embedding_status == EmbeddingStatus.FAILED.value
    assert completed_chunk.embedding_status == EmbeddingStatus.COMPLETED.value


@pytest.mark.parametrize(
    ("source_status", "expected_status"),
    [
        (DocumentStatus.PARSING, DocumentStatus.PARSE_FAILED),
        (DocumentStatus.CHUNKING, DocumentStatus.CHUNK_FAILED),
    ],
)
def test_recovery_resets_interrupted_document_state(
    db: Session,
    source_status: DocumentStatus,
    expected_status: DocumentStatus,
) -> None:
    """验证解析/切片中断状态可以恢复为已有重试入口识别的失败状态。"""
    document = create_document(
        db=db,
        status=source_status,
        filename=f"recover-{source_status.value}.txt",
    )
    job = ProcessingJob(
        document_id=document.id,
        job_type=ProcessingJobType.FULL_PIPELINE.value,
        status=ProcessingJobStatus.RUNNING.value,
        stage=ProcessingJobStage.PARSING.value,
        progress=10,
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    recovery = ProcessingJobRecoveryService(
        document_repository=DocumentRepository(),
        document_chunk_repository=DocumentChunkRepository(),
    )
    recovery.prepare_for_resume(db=db, job=job)
    db.refresh(document)
    assert document.status == expected_status.value


def test_retry_policy_only_retries_transient_failures() -> None:
    """验证永久业务异常不会被 Celery 无意义重复执行。"""
    policy = ProcessingJobRetryPolicy(base_delay_seconds=2, max_delay_seconds=10)

    class ServiceUnavailableError(RuntimeError):
        status_code = 503

    assert policy.should_retry(TimeoutError("timeout")) is True
    assert policy.should_retry(ServiceUnavailableError("temporary")) is True
    assert policy.should_retry(ValueError("bad pdf")) is False
    assert policy.retry_delay_seconds(0) == 2
    assert policy.retry_delay_seconds(1) == 4
    assert policy.retry_delay_seconds(3) == 10


def test_claimed_retry_from_indexing_does_not_regress_stage(db: Session) -> None:
    """验证 Qdrant 阶段重试不会把 ProcessingJob 阶段倒退到 embedding。"""
    document = create_document(
        db=db,
        status=DocumentStatus.CHUNKED,
        filename="resume-indexing.txt",
    )
    calls: list[str] = []
    executor = build_processing_job_executor(
        document_processing_service=FakeDocumentProcessingService(calls=calls),
        embedding_service=FakeEmbeddingService(processed_count=0, calls=calls),
        vector_index_service=FakeVectorIndexService(calls=calls),
    )
    job = executor.processing_job_service.create_job(
        db=db,
        document_id=document.id,
        job_type=ProcessingJobType.FULL_PIPELINE,
    )
    executor.processing_job_service.start_job(db=db, job_id=job.id)
    executor.processing_job_service.update_stage(
        db=db,
        job_id=job.id,
        stage=ProcessingJobStage.EMBEDDING,
        progress=60,
    )
    executor.processing_job_service.update_stage(
        db=db,
        job_id=job.id,
        stage=ProcessingJobStage.INDEXING,
        progress=85,
    )
    document.status = DocumentStatus.COMPLETED.value
    db.commit()

    executor.execute_claimed_job(db=db, job_id=job.id)

    saved_job = executor.processing_job_service.get_job(db=db, job_id=job.id)
    assert saved_job.status == ProcessingJobStatus.SUCCEEDED.value
    assert saved_job.stage == ProcessingJobStage.COMPLETED.value
    assert saved_job.progress == 100
    assert calls[-1] == "vector_index"


class FailingWorkerRunner(FakeWorkerRunner):
    """Celery 重试测试使用的失败 Runner。"""

    def __init__(self, error: Exception) -> None:
        super().__init__()
        self.error = error

    def run_worker(self, job_id: int, force_resume: bool = False) -> bool:
        self.received_job_id = job_id
        raise self.error


class RetryScheduled(Exception):
    """替代 Celery self.retry 的测试信号。"""


def test_celery_task_retries_transient_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """验证瞬时异常释放租约并由 Celery 安排重试。"""
    fake_runner = FailingWorkerRunner(TimeoutError("temporary timeout"))
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        "app.tasks.processing_job.get_processing_job_runner",
        lambda: fake_runner,
    )
    monkeypatch.setattr(retry_policy, "should_retry", lambda exc: True)
    monkeypatch.setattr(retry_policy, "retry_delay_seconds", lambda count: 2)

    def fake_retry(*args, **kwargs):
        captured.update(kwargs)
        raise RetryScheduled()

    monkeypatch.setattr(execute_processing_job, "retry", fake_retry)

    with pytest.raises(RetryScheduled):
        execute_processing_job.run(31)

    assert fake_runner.received_job_id == 31
    assert fake_runner.released_job_id == 31
    assert fake_runner.failed_job_id is None
    assert captured["countdown"] == 2


def test_celery_task_marks_non_retryable_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """验证永久业务异常不反复重试，直接结束 ProcessingJob。"""
    error = ValueError("invalid pdf")
    fake_runner = FailingWorkerRunner(error)
    monkeypatch.setattr(
        "app.tasks.processing_job.get_processing_job_runner",
        lambda: fake_runner,
    )
    monkeypatch.setattr(retry_policy, "should_retry", lambda exc: False)

    with pytest.raises(ValueError, match="invalid pdf"):
        execute_processing_job.run(32)

    assert fake_runner.released_job_id is None
    assert fake_runner.failed_job_id == 32


def test_celery_task_skips_existing_worker_lease(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证普通重复投递遇到有效租约时直接跳过，不并发执行。"""
    fake_runner = FailingWorkerRunner(ProcessingJobAlreadyClaimedError(7))
    monkeypatch.setattr(
        "app.tasks.processing_job.get_processing_job_runner",
        lambda: fake_runner,
    )

    execute_processing_job.run(33)

    assert fake_runner.received_job_id == 33
    assert fake_runner.released_job_id is None
    assert fake_runner.failed_job_id is None



def test_worker_claim_force_resumes_redelivered_job(db: Session) -> None:
    """验证 Broker 明确 redeliver 时可立即接管旧 Worker 的仍有效租约。"""
    document = create_document(
        db=db,
        status=DocumentStatus.UPLOADED,
        filename="redelivered-lease.txt",
    )
    service = ProcessingJobService(
        document_repository=DocumentRepository(),
        processing_job_repository=ProcessingJobRepository(),
        lease_seconds=600,
    )
    job = service.create_job(
        db=db,
        document_id=document.id,
        job_type=ProcessingJobType.FULL_PIPELINE,
    )
    service.claim_job_for_worker(db=db, job_id=job.id)

    claim = service.claim_job_for_worker(
        db=db,
        job_id=job.id,
        force_resume=True,
    )
    assert claim is not None
    assert claim.resumed is True
    assert service.get_job(db=db, job_id=job.id).attempt_count == 2
