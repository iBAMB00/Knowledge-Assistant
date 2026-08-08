import logging
from collections.abc import Callable

from sqlalchemy.orm import Session

from app.constants.document_status import DocumentStatus
from app.constants.processing_job_stage import ProcessingJobStage
from app.constants.processing_job_status import ProcessingJobStatus
from app.constants.processing_job_type import ProcessingJobType
from app.repositories.document_repository import DocumentRepository
from app.schemas.document_response import DocumentResponse
from app.services.document_processing_service import DocumentProcessingService
from app.services.embedding_service import EmbeddingService
from app.services.processing_job_service import (
    InvalidProcessingJobError,
    ProcessingJobService,
)
from app.services.vector_index_service import VectorIndexService

logger = logging.getLogger(__name__)


class ProcessingJobExecutor:
    """
    文档处理任务执行器。

    提供两类入口：
    - process_document / embed_document：
      兼容同步接口，创建任务后立即执行
    - execute_job：
      执行已经创建的任务，供后台Runner或未来Worker调用

    不负责：
    - HTTP协议转换
    - 后台线程或消息队列
    - 创建数据库Session
    """

    DOCUMENT_PARSING_PROGRESS = 10
    DOCUMENT_CHUNKING_PROGRESS = 60

    PIPELINE_PARSING_PROGRESS = 10
    PIPELINE_CHUNKING_PROGRESS = 35
    PIPELINE_EMBEDDING_PROGRESS = 60
    PIPELINE_INDEXING_PROGRESS = 85

    EMBEDDING_PROGRESS = 10
    EMBEDDING_INDEXING_PROGRESS = 80

    FINALIZING_PROGRESS = 95

    STAGE_ORDER = {
        ProcessingJobStage.QUEUED: 0,
        ProcessingJobStage.PARSING: 1,
        ProcessingJobStage.CHUNKING: 2,
        ProcessingJobStage.EMBEDDING: 3,
        ProcessingJobStage.INDEXING: 4,
        ProcessingJobStage.FINALIZING: 5,
        ProcessingJobStage.COMPLETED: 6,
    }

    DOCUMENT_PROCESSING_NOOP_STATUSES = frozenset({
        DocumentStatus.CHUNKED,
        DocumentStatus.EMBEDDING,
        DocumentStatus.EMBEDDING_FAILED,
        DocumentStatus.COMPLETED,
    })

    def __init__(
        self,
        document_repository: DocumentRepository,
        processing_job_service: ProcessingJobService,
        document_processing_service: DocumentProcessingService,
        embedding_service: EmbeddingService,
        vector_index_service: VectorIndexService | None = None,
    ) -> None:
        self.document_repository = document_repository
        self.processing_job_service = processing_job_service
        self.document_processing_service = document_processing_service
        self.embedding_service = embedding_service
        self.vector_index_service = vector_index_service

    def process_document(
        self,
        db: Session,
        document_id: int,
    ) -> DocumentResponse:
        """
        创建并同步执行文档解析切片任务。

        文档已经完成解析切片时保持幂等，
        直接返回业务结果，不创建空任务。
        """

        current_status = self._get_document_status(
            db=db,
            document_id=document_id,
        )

        if current_status in self.DOCUMENT_PROCESSING_NOOP_STATUSES:
            return self.document_processing_service.process_document(
                db=db,
                document_id=document_id,
            )

        result = self._create_and_execute_job(
            db=db,
            document_id=document_id,
            job_type=ProcessingJobType.DOCUMENT_PROCESSING,
        )

        if not isinstance(result, DocumentResponse):
            raise RuntimeError(
                "unexpected document processing result"
            )

        return result

    def embed_document(
        self,
        db: Session,
        document_id: int,
        batch_size: int = 100,
    ) -> int:
        """
        创建并同步执行文档向量化任务。

        completed文档未启用外部索引时保持幂等，
        启用外部索引时创建任务，用于补建或重建索引。
        """

        current_status = self._get_document_status(
            db=db,
            document_id=document_id,
        )

        if (
            current_status == DocumentStatus.COMPLETED
            and self.vector_index_service is None
        ):
            return self.embedding_service.process_document(
                db=db,
                document_id=document_id,
                batch_size=batch_size,
            )

        result = self._create_and_execute_job(
            db=db,
            document_id=document_id,
            job_type=ProcessingJobType.EMBEDDING,
            batch_size=batch_size,
        )

        if not isinstance(result, int):
            raise RuntimeError(
                "unexpected embedding result"
            )

        return result

    def execute_job(
        self,
        db: Session,
        job_id: int,
        batch_size: int = 100,
    ) -> DocumentResponse | int:
        """
        执行已经创建的pending任务。

        同步兼容入口仍由Executor负责最终失败状态。
        """

        job = self.processing_job_service.start_job(
            db=db,
            job_id=job_id,
        )

        return self._execute_running_job(
            db=db,
            job_id=job.id,
            document_id=job.document_id,
            job_type_value=job.job_type,
            batch_size=batch_size,
            terminalize_failure=True,
        )

    def execute_claimed_job(
        self,
        db: Session,
        job_id: int,
        batch_size: int = 100,
    ) -> DocumentResponse | int:
        """
        执行 Worker 已领取的running任务。

        Worker瞬时失败是否重试由Celery层决定，
        因此本入口不立即把ProcessingJob写成failed。
        """

        job = self.processing_job_service.get_job(
            db=db,
            job_id=job_id,
        )

        if (
            ProcessingJobStatus(job.status)
            != ProcessingJobStatus.RUNNING
        ):
            raise InvalidProcessingJobError(
                "worker job must be running before execution"
            )

        return self._execute_running_job(
            db=db,
            job_id=job.id,
            document_id=job.document_id,
            job_type_value=job.job_type,
            batch_size=batch_size,
            terminalize_failure=False,
        )

    def _execute_running_job(
        self,
        db: Session,
        job_id: int,
        document_id: int,
        job_type_value: str,
        batch_size: int,
        terminalize_failure: bool,
    ) -> DocumentResponse | int:
        """执行已经进入running状态的任务主体。"""

        try:
            result = self._execute_business(
                db=db,
                job_id=job_id,
                document_id=document_id,
                job_type=ProcessingJobType(job_type_value),
                batch_size=batch_size,
            )

            self._advance_stage(
                db=db,
                job_id=job_id,
                stage=ProcessingJobStage.FINALIZING,
                progress=self.FINALIZING_PROGRESS,
            )

            self.processing_job_service.succeed_job(
                db=db,
                job_id=job_id,
            )

            return result

        except Exception as exc:
            logger.error(
                "Processing job execution failed: "
                "job_id=%s, job_type=%s, error_type=%s",
                job_id,
                job_type_value,
                type(exc).__name__,
            )

            db.rollback()

            if terminalize_failure:
                self._safe_fail_job(
                    db=db,
                    job_id=job_id,
                    job_type_value=job_type_value,
                )

            raise

    def _create_and_execute_job(
        self,
        db: Session,
        document_id: int,
        job_type: ProcessingJobType,
        batch_size: int = 100,
    ) -> DocumentResponse | int:
        """
        为同步兼容入口创建任务并立即执行。
        """

        job = self.processing_job_service.create_job(
            db=db,
            document_id=document_id,
            job_type=job_type,
        )

        return self.execute_job(
            db=db,
            job_id=job.id,
            batch_size=batch_size,
        )

    def _execute_business(
        self,
        db: Session,
        job_id: int,
        document_id: int,
        job_type: ProcessingJobType,
        batch_size: int,
    ) -> DocumentResponse | int:
        """
        根据任务类型调用底层业务服务。

        本方法不得调用process_document或embed_document，
        防止再次创建ProcessingJob。
        """

        if job_type == ProcessingJobType.DOCUMENT_PROCESSING:
            return self._execute_document_processing(
                db=db,
                job_id=job_id,
                document_id=document_id,
            )

        if job_type == ProcessingJobType.EMBEDDING:
            return self._execute_embedding(
                db=db,
                job_id=job_id,
                document_id=document_id,
                batch_size=batch_size,
            )

        if job_type == ProcessingJobType.FULL_PIPELINE:
            return self._execute_full_pipeline(
                db=db,
                job_id=job_id,
                document_id=document_id,
                batch_size=batch_size,
            )

        raise InvalidProcessingJobError(
            "unsupported processing job type: "
            f"{job_type.value}"
        )

    def _execute_document_processing(
        self,
        db: Session,
        job_id: int,
        document_id: int,
    ) -> DocumentResponse:
        """
        执行文档解析与切片任务。

        真实阶段由DocumentProcessingService的文档状态回调驱动。
        """

        return self.document_processing_service.process_document(
            db=db,
            document_id=document_id,
            status_callback=self._build_document_status_callback(
                db=db,
                job_id=job_id,
                job_type=ProcessingJobType.DOCUMENT_PROCESSING,
            ),
        )

    def _execute_embedding(
        self,
        db: Session,
        job_id: int,
        document_id: int,
        batch_size: int,
    ) -> int:
        """
        执行文档向量化和可选的外部索引同步。
        """

        return self._embed_and_index_document(
            db=db,
            job_id=job_id,
            document_id=document_id,
            batch_size=batch_size,
            embedding_progress=self.EMBEDDING_PROGRESS,
            indexing_progress=(
                self.EMBEDDING_INDEXING_PROGRESS
            ),
        )

    def _execute_full_pipeline(
        self,
        db: Session,
        job_id: int,
        document_id: int,
        batch_size: int,
    ) -> int:
        """
        在同一个父任务中顺序执行解析、切片、向量化和索引。
        """

        self.document_processing_service.process_document(
            db=db,
            document_id=document_id,
            status_callback=self._build_document_status_callback(
                db=db,
                job_id=job_id,
                job_type=ProcessingJobType.FULL_PIPELINE,
            ),
        )

        return self._embed_and_index_document(
            db=db,
            job_id=job_id,
            document_id=document_id,
            batch_size=batch_size,
            embedding_progress=self.PIPELINE_EMBEDDING_PROGRESS,
            indexing_progress=self.PIPELINE_INDEXING_PROGRESS,
        )

    def _embed_and_index_document(
        self,
        db: Session,
        job_id: int,
        document_id: int,
        batch_size: int,
        embedding_progress: int,
        indexing_progress: int,
    ) -> int:
        """
        生成并保存文档向量，随后同步外部向量索引。
        """

        self._advance_stage(
            db=db,
            job_id=job_id,
            stage=ProcessingJobStage.EMBEDDING,
            progress=embedding_progress,
        )

        processed_count = self.embedding_service.process_document(
            db=db,
            document_id=document_id,
            batch_size=batch_size,
        )

        if self.vector_index_service is not None:
            self._advance_stage(
                db=db,
                job_id=job_id,
                stage=ProcessingJobStage.INDEXING,
                progress=indexing_progress,
            )

            self.vector_index_service.index_document(
                db=db,
                document_id=document_id,
            )

        return processed_count

    def _build_document_status_callback(
        self,
        db: Session,
        job_id: int,
        job_type: ProcessingJobType,
    ) -> Callable[[DocumentStatus], None]:
        """
        构造文档状态到任务阶段的映射回调。
        """

        def update_job_stage(
            document_status: DocumentStatus,
        ) -> None:
            if document_status == DocumentStatus.PARSING:
                progress = (
                    self.DOCUMENT_PARSING_PROGRESS
                    if job_type
                    == ProcessingJobType.DOCUMENT_PROCESSING
                    else self.PIPELINE_PARSING_PROGRESS
                )

                self._advance_stage(
                    db=db,
                    job_id=job_id,
                    stage=ProcessingJobStage.PARSING,
                    progress=progress,
                )
                return

            if document_status == DocumentStatus.CHUNKING:
                progress = (
                    self.DOCUMENT_CHUNKING_PROGRESS
                    if job_type
                    == ProcessingJobType.DOCUMENT_PROCESSING
                    else self.PIPELINE_CHUNKING_PROGRESS
                )

                self._advance_stage(
                    db=db,
                    job_id=job_id,
                    stage=ProcessingJobStage.CHUNKING,
                    progress=progress,
                )

        return update_job_stage

    def _advance_stage(
        self,
        db: Session,
        job_id: int,
        stage: ProcessingJobStage,
        progress: int,
    ) -> None:
        """
        单调推进任务阶段。

        Worker重试已经越过某阶段时直接跳过，
        避免indexing重试重新把任务阶段写回embedding。
        """

        job = self.processing_job_service.get_job(
            db=db,
            job_id=job_id,
        )
        current_stage = ProcessingJobStage(job.stage)

        if (
            self.STAGE_ORDER[current_stage]
            > self.STAGE_ORDER[stage]
        ):
            return

        self.processing_job_service.update_stage(
            db=db,
            job_id=job_id,
            stage=stage,
            progress=max(job.progress, progress),
        )

    def _safe_fail_job(
        self,
        db: Session,
        job_id: int,
        job_type_value: str,
    ) -> None:
        """
        安全记录任务失败状态。

        失败状态保存异常不能覆盖原始业务异常。
        """

        error_message = self._build_public_error_message(
            job_type_value=job_type_value,
        )

        try:
            self.processing_job_service.fail_job(
                db=db,
                job_id=job_id,
                error_message=error_message,
            )

        except Exception as exc:
            db.rollback()

            logger.error(
                "Failed to mark processing job failed: "
                "job_id=%s, error_type=%s",
                job_id,
                type(exc).__name__,
            )

    def _get_document_status(
        self,
        db: Session,
        document_id: int,
    ) -> DocumentStatus:
        """查询并转换文档状态。"""

        document = self.document_repository.find_by_id(
            db=db,
            document_id=document_id,
        )

        if document is None:
            raise ValueError("document not found")

        return DocumentStatus(document.status)

    @staticmethod
    def _build_public_error_message(
        job_type_value: str,
    ) -> str:
        """
        返回不包含底层异常详情的公开错误摘要。
        """

        if (
            job_type_value
            == ProcessingJobType.DOCUMENT_PROCESSING.value
        ):
            return "文档解析或切片失败"

        if job_type_value == ProcessingJobType.EMBEDDING.value:
            return "文档向量化失败"

        if (
            job_type_value
            == ProcessingJobType.FULL_PIPELINE.value
        ):
            return "文档完整处理失败"

        return "文档处理任务失败"
