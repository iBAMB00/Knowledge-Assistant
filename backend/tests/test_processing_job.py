from collections.abc import Iterator
from datetime import datetime

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.api.knowledge import get_processing_job_runner, router
from app.api.processing_job import router as processing_job_router
from app.constants.document_status import DocumentStatus
from app.constants.processing_job_status import ProcessingJobStatus
from app.constants.processing_job_type import ProcessingJobType
from app.core.database import get_db
from app.models.database.document import Document
from app.models.database.processing_job import ProcessingJob
from app.repositories.document_repository import DocumentRepository
from app.repositories.processing_job_repository import ProcessingJobRepository
from app.schemas.document_response import DocumentResponse
from app.services.processing_job_executor import ProcessingJobExecutor
from app.services.processing_job_runner import ProcessingJobRunner
from app.services.processing_job_service import (
    ActiveProcessingJobError,
    InvalidProcessingJobError,
    ProcessingJobService,
)
from app.services.processing_job_service import ProcessingJobNotFoundError
from app.services.status_machine import InvalidStatusTransitionError

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
    ) -> DocumentResponse:
        self.call_count += 1

        if self.calls is not None:
            self.calls.append("document_processing")

        if self.error is not None:
            raise self.error

        return DocumentResponse(
            id=document_id,
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

class FakeProcessingJobRunner:
    """
    API测试使用的任务Runner。

    不执行真实解析、切片和向量化，
    只记录接口提交的任务ID。
    """

    def __init__(self) -> None:
        self.received_job_id: int | None = None

    def run(self, job_id: int) -> None:
        self.received_job_id = job_id

def build_processing_job_executor(
    document_processing_service: (
        FakeDocumentProcessingService | None
    ) = None,
    embedding_service: (
        FakeEmbeddingService | None
    ) = None,
) -> ProcessingJobExecutor:
    """
    创建任务执行器。
    """

    document_repository = DocumentRepository()

    processing_job_service = (
        ProcessingJobService(
            document_repository=(
                document_repository
            ),
            processing_job_repository=(
                ProcessingJobRepository()
            ),
        )
    )

    return ProcessingJobExecutor(
        document_repository=document_repository,
        processing_job_service=(
            processing_job_service
        ),
        document_processing_service=(
            document_processing_service
            or FakeDocumentProcessingService()
        ),  # type: ignore[arg-type]
        embedding_service=(
            embedding_service
            or FakeEmbeddingService()
        ),  # type: ignore[arg-type]
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

    runner.run(job_id=11)

    assert fake_session.rolled_back is True
    assert fake_session.closed is True

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
) -> Iterator[tuple[TestClient, FakeProcessingJobRunner]]:
    """创建后台任务接口测试客户端。"""

    app = FastAPI()
    app.include_router(router)
    app.include_router(processing_job_router)

    fake_runner = FakeProcessingJobRunner()

    def override_get_db():
        yield db

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_processing_job_runner] = lambda: fake_runner

    with TestClient(app) as client:
        yield client, fake_runner

def test_create_document_processing_job_api(
    db: Session,
    processing_job_client: tuple[TestClient, FakeProcessingJobRunner],
) -> None:
    """验证接口创建任务并交给后台Runner。"""

    client, fake_runner = processing_job_client

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
    assert body["progress"] == 0
    assert fake_runner.received_job_id == body["id"]

def test_create_processing_job_returns_404(
    processing_job_client: tuple[TestClient, FakeProcessingJobRunner],
) -> None:
    """验证文档不存在时不提交后台任务。"""

    client, fake_runner = processing_job_client

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
    assert fake_runner.received_job_id is None

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
    progress_updates: list[int] = []

    document_processing_service = (
        FakeDocumentProcessingService(
            calls=calls,
        )
    )

    embedding_service = FakeEmbeddingService(
        calls=calls,
    )

    executor = build_processing_job_executor(
        document_processing_service=(
            document_processing_service
        ),
        embedding_service=embedding_service,
    )

    original_update_progress = (
        executor
        .processing_job_service
        .update_progress
    )

    def record_progress(
        db: Session,
        job_id: int,
        progress: int,
    ) -> ProcessingJob:
        """记录执行器提交的阶段进度。"""

        progress_updates.append(progress)

        return original_update_progress(
            db=db,
            job_id=job_id,
            progress=progress,
        )

    monkeypatch.setattr(
        executor.processing_job_service,
        "update_progress",
        record_progress,
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
    ]

    assert progress_updates == [
        10,
        60,
        90,
    ]

    assert (
        document_processing_service.call_count
        == 1
    )

    assert embedding_service.call_count == 1

    assert saved_job.status == (
        ProcessingJobStatus.SUCCEEDED.value
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
        FakeProcessingJobRunner,
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
        FakeProcessingJobRunner,
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
        FakeProcessingJobRunner,
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
        FakeProcessingJobRunner,
    ],
) -> None:
    """
    验证同一文档已有活动任务时，
    接口拒绝创建第二个任务。
    """

    client, fake_runner = processing_job_client

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
        fake_runner.received_job_id
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
        FakeProcessingJobRunner,
    ],
) -> None:
    """
    验证旧任务失败后保留历史，
    同时允许接口创建新的重试任务。
    """

    client, fake_runner = processing_job_client

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
    assert body["progress"] == 0
    assert body["error_message"] is None

    assert fake_runner.received_job_id == body["id"]

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
        FakeProcessingJobRunner,
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
    assert body[0]["progress"] == 0
    assert body[0]["error_message"] is None

    assert body[1]["document_id"] == document.id

    assert (
        body[1]["job_type"]
        == "document_processing"
    )

    assert body[1]["status"] == "failed"

    assert body[1]["error_message"] == (
        "文档解析或切片失败"
    )

def test_list_document_processing_jobs_returns_empty_list(
    db: Session,
    processing_job_client: tuple[
        TestClient,
        FakeProcessingJobRunner,
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
        FakeProcessingJobRunner,
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