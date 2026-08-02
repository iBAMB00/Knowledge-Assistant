import logging

from sqlalchemy.orm import Session

from app.constants.document_status import DocumentStatus
from app.constants.processing_job_type import ProcessingJobType
from app.repositories.document_repository import DocumentRepository
from app.schemas.document_response import DocumentResponse
from app.services.document_processing_service import DocumentProcessingService
from app.services.embedding_service import EmbeddingService
from app.services.processing_job_service import ProcessingJobService, InvalidProcessingJobError



logger = logging.getLogger(__name__)


class ProcessingJobExecutor:
    """
    文档处理任务执行器。

    负责：
    - 创建ProcessingJob
    - 将任务标记为running
    - 调用真实业务Service
    - 将任务标记为succeeded或failed

    不负责：
    - HTTP协议转换
    - 后台线程或消息队列
    - 创建数据库Session
    """

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
        document_processing_service: (
            DocumentProcessingService
        ),
        embedding_service: EmbeddingService,
    ) -> None:
        self.document_repository = document_repository
        self.processing_job_service = (
            processing_job_service
        )
        self.document_processing_service = (
            document_processing_service
        )
        self.embedding_service = embedding_service

    def process_document(
        self,
        db: Session,
        document_id: int,
    ) -> DocumentResponse:
        """
        创建并同步执行文档解析切片任务。

        文档已经完成解析切片时，沿用原有幂等语义，
        不创建没有实际工作的任务记录。
        """

        current_status = self._get_document_status(
            db=db,
            document_id=document_id,
        )

        if (
            current_status
            in self.DOCUMENT_PROCESSING_NOOP_STATUSES
        ):
            return (
                self.document_processing_service
                .process_document(
                    db=db,
                    document_id=document_id,
                )
            )

        job = self.processing_job_service.create_job(
            db=db,
            document_id=document_id,
            job_type=(
                ProcessingJobType
                .DOCUMENT_PROCESSING
            ),
        )

        result = self.execute_job(
            db=db,
            job_id=job.id,
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

        completed文档保持原有幂等语义，
        不额外创建任务记录。
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

        job = self.processing_job_service.create_job(
            db=db,
            document_id=document_id,
            job_type=ProcessingJobType.EMBEDDING,
        )

        result = self.execute_job(
            db=db,
            job_id=job.id,
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

        未来Worker可以直接调用这个方法，
        不需要经过FastAPI Router。
        """

        job = self.processing_job_service.start_job(
            db=db,
            job_id=job_id,
        )

        document_id = job.document_id
        job_type_value = job.job_type

        try:
            job_type = ProcessingJobType(
                job_type_value
            )

            if job_type == ProcessingJobType.DOCUMENT_PROCESSING:
                result = self.document_processing_service.process_document(
                    db=db,
                    document_id=job.document_id,
                )

            elif job_type == ProcessingJobType.EMBEDDING:
                result = self.embedding_service.process_document(
                    db=db,
                    document_id=job.document_id,
                )
            
            elif job_type == ProcessingJobType.FULL_PIPELINE:
                result = self._execute_full_pipeline(
                    db=db,
                    document_id=job.document_id,
                )

            else:
                raise InvalidProcessingJobError(
                    f"unsupported processing job type: {job_type_value}"
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

            self._safe_fail_job(
                db=db,
                job_id=job_id,
                job_type_value=job_type_value,
            )

            raise

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

        error_message = (
            self._build_public_error_message(
                job_type_value=job_type_value,
            )
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
        """
        查询并转换文档状态。
        """

        document = (
            self.document_repository.find_by_id(
                db=db,
                document_id=document_id,
            )
        )

        if document is None:
            raise ValueError(
                "document not found"
            )

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
            == ProcessingJobType
            .DOCUMENT_PROCESSING
            .value
        ):
            return "文档解析或切片失败"

        if (
            job_type_value
            == ProcessingJobType
            .EMBEDDING
            .value
        ):
            return "文档向量化失败"

        return "文档处理任务失败"
    
    def _execute_full_pipeline(
        self,
        db: Session,
        document_id: int,
    ) -> int:
        """
        顺序执行文档解析切片和向量化。

        Returns:
            本次成功向量化的Chunk数量。
        """

        self.document_processing_service.process_document(
            db=db,
            document_id=document_id,
        )

        return self.embedding_service.process_document(
            db=db,
            document_id=document_id,
        )
    