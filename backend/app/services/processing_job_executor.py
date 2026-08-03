import logging

from sqlalchemy.orm import Session

from app.constants.document_status import DocumentStatus
from app.constants.processing_job_type import ProcessingJobType
from app.repositories.document_repository import DocumentRepository
from app.schemas.document_response import DocumentResponse
from app.services.document_processing_service import DocumentProcessingService
from app.services.embedding_service import EmbeddingService
from app.services.processing_job_service import (
    InvalidProcessingJobError,
    ProcessingJobService,
)


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

    INITIAL_PROGRESS = 10
    PIPELINE_DOCUMENT_COMPLETED_PROGRESS = 60
    BUSINESS_COMPLETED_PROGRESS = 90

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
    ) -> None:
        self.document_repository = document_repository
        self.processing_job_service = processing_job_service
        self.document_processing_service = document_processing_service
        self.embedding_service = embedding_service

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

        completed文档保持幂等，
        直接返回业务结果，不创建空任务。
        """

        current_status = self._get_document_status(
            db=db,
            document_id=document_id,
        )

        if current_status == DocumentStatus.COMPLETED:
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

        该方法绝不创建新任务，只消费指定job_id。

        执行过程中保存粗粒度阶段进度，
        成功时由succeed_job统一更新为100。
        """

        job = self.processing_job_service.start_job(
            db=db,
            job_id=job_id,
        )

        job_type_value = job.job_type

        try:
            result = self._execute_business(
                db=db,
                job_id=job_id,
                document_id=job.document_id,
                job_type=ProcessingJobType(
                    job_type_value
                ),
                batch_size=batch_size,
            )

            self.processing_job_service.succeed_job(
                db=db,
                job_id=job_id,
            )

            return result

        except Exception as exc:
            logger.error(
                "Processing job execution failed: "
                "job_id=%s, job_type=%s, "
                "error_type=%s",
                job_id,
                job_type_value,
                type(exc).__name__,
            )

            # 保证Session脱离可能存在的失败事务，
            # 后续才能可靠写入Job失败状态。
            db.rollback()

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

        if (
            job_type
            == ProcessingJobType.DOCUMENT_PROCESSING
        ):
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

        进度语义：
        - 10：开始处理
        - 90：业务处理完成，等待任务收尾
        """

        self.processing_job_service.update_progress(
            db=db,
            job_id=job_id,
            progress=self.INITIAL_PROGRESS,
        )

        result = (
            self.document_processing_service
            .process_document(
                db=db,
                document_id=document_id,
            )
        )

        self.processing_job_service.update_progress(
            db=db,
            job_id=job_id,
            progress=self.BUSINESS_COMPLETED_PROGRESS,
        )

        return result

    def _execute_embedding(
        self,
        db: Session,
        job_id: int,
        document_id: int,
        batch_size: int,
    ) -> int:
        """
        执行文档向量化任务。

        进度语义：
        - 10：开始向量化
        - 90：向量化完成，等待任务收尾
        """

        self.processing_job_service.update_progress(
            db=db,
            job_id=job_id,
            progress=self.INITIAL_PROGRESS,
        )

        processed_count = (
            self.embedding_service.process_document(
                db=db,
                document_id=document_id,
                batch_size=batch_size,
            )
        )

        self.processing_job_service.update_progress(
            db=db,
            job_id=job_id,
            progress=self.BUSINESS_COMPLETED_PROGRESS,
        )

        return processed_count

    def _execute_full_pipeline(
        self,
        db: Session,
        job_id: int,
        document_id: int,
        batch_size: int,
    ) -> int:
        """
        在同一个父任务中顺序执行解析切片和向量化。

        进度语义：
        - 10：开始完整处理
        - 60：解析与切片完成
        - 90：向量化完成，等待任务收尾
        """

        self.processing_job_service.update_progress(
            db=db,
            job_id=job_id,
            progress=self.INITIAL_PROGRESS,
        )

        self.document_processing_service.process_document(
            db=db,
            document_id=document_id,
        )

        self.processing_job_service.update_progress(
            db=db,
            job_id=job_id,
            progress=(
                self
                .PIPELINE_DOCUMENT_COMPLETED_PROGRESS
            ),
        )

        processed_count = (
            self.embedding_service.process_document(
                db=db,
                document_id=document_id,
                batch_size=batch_size,
            )
        )

        self.processing_job_service.update_progress(
            db=db,
            job_id=job_id,
            progress=self.BUSINESS_COMPLETED_PROGRESS,
        )

        return processed_count

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
            raise ValueError(
                "document not found"
            )

        return DocumentStatus(
            document.status
        )

    @staticmethod
    def _build_public_error_message(
        job_type_value: str,
    ) -> str:
        """
        返回不包含底层异常详情的公开错误摘要。
        """

        if (
            job_type_value
            == ProcessingJobType
            .DOCUMENT_PROCESSING
            .value
        ):
            return "文档解析或切片失败"

        if (
            job_type_value
            == ProcessingJobType.EMBEDDING.value
        ):
            return "文档向量化失败"

        if (
            job_type_value
            == ProcessingJobType.FULL_PIPELINE.value
        ):
            return "文档完整处理失败"

        return "文档处理任务失败"