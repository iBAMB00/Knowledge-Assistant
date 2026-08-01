from datetime import datetime

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.constants.document_status import (
    DocumentStatus,
)
from app.constants.processing_job_status import (
    ProcessingJobStatus,
)
from app.constants.processing_job_type import (
    ProcessingJobType,
)
from app.models.database.processing_job import (
    ProcessingJob,
)
from app.repositories.document_repository import (
    DocumentRepository,
)
from app.repositories.processing_job_repository import (
    ProcessingJobRepository,
)
from app.services.status_machine import (
    StatusMachine,
)


class ProcessingJobNotFoundError(ValueError):
    """
    指定任务不存在。
    """


class ActiveProcessingJobError(ValueError):
    """
    文档已经存在活动任务。
    """


class InvalidProcessingJobError(ValueError):
    """
    当前文档或任务状态不允许执行操作。
    """


class ProcessingJobService:
    """
    文档处理任务管理服务。

    负责：
    - 创建任务
    - 防止重复活动任务
    - 管理任务状态和进度
    - 管理事务边界

    暂时不负责真正执行文档处理任务。
    """

    MAX_ERROR_MESSAGE_LENGTH = 500

    def __init__(
        self,
        document_repository: DocumentRepository,
        processing_job_repository: (
            ProcessingJobRepository
        ),
    ) -> None:
        self.document_repository = (
            document_repository
        )
        self.processing_job_repository = (
            processing_job_repository
        )

    def create_job(
        self,
        db: Session,
        document_id: int,
        job_type: ProcessingJobType,
    ) -> ProcessingJob:
        """
        为指定文档创建待执行任务。
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

        self._validate_document_status(
            document_status=DocumentStatus(
                document.status
            ),
            job_type=job_type,
        )

        active_job = (
            self.processing_job_repository
            .find_active_by_document_id(
                db=db,
                document_id=document_id,
            )
        )

        if active_job is not None:
            raise ActiveProcessingJobError(
                "document already has an active job"
            )

        job = ProcessingJob(
            document_id=document_id,
            job_type=job_type.value,
            status=(
                ProcessingJobStatus.PENDING.value
            ),
            progress=0,
        )

        try:
            self.processing_job_repository.create(
                db=db,
                job=job,
            )

            db.commit()
            db.refresh(job)

            return job

        except IntegrityError as exc:
            db.rollback()

            raise ActiveProcessingJobError(
                "document already has an active job"
            ) from exc

        except Exception:
            db.rollback()
            raise

    def start_job(
        self,
        db: Session,
        job_id: int,
    ) -> ProcessingJob:
        """
        将pending任务转换为running。
        """

        job = self._get_job(
            db=db,
            job_id=job_id,
        )

        StatusMachine.transition_processing_job(
            job=job,
            target_status=(
                ProcessingJobStatus.RUNNING
            ),
        )

        job.started_at = datetime.utcnow()
        job.finished_at = None
        job.error_message = None

        db.commit()
        db.refresh(job)

        return job

    def update_progress(
        self,
        db: Session,
        job_id: int,
        progress: int,
    ) -> ProcessingJob:
        """
        更新运行中任务进度。

        running状态下进度只能为0到99；
        成功时统一设置为100。
        """

        if progress < 0 or progress >= 100:
            raise ValueError(
                "running job progress must be "
                "between 0 and 99"
            )

        job = self._get_job(
            db=db,
            job_id=job_id,
        )

        if (
            ProcessingJobStatus(job.status)
            != ProcessingJobStatus.RUNNING
        ):
            raise InvalidProcessingJobError(
                "only running job can update progress"
            )

        job.progress = progress

        db.commit()
        db.refresh(job)

        return job

    def succeed_job(
        self,
        db: Session,
        job_id: int,
    ) -> ProcessingJob:
        """
        将运行中的任务标记为成功。
        """

        job = self._get_job(
            db=db,
            job_id=job_id,
        )

        StatusMachine.transition_processing_job(
            job=job,
            target_status=(
                ProcessingJobStatus.SUCCEEDED
            ),
        )

        job.progress = 100
        job.finished_at = datetime.utcnow()
        job.error_message = None

        db.commit()
        db.refresh(job)

        return job

    def fail_job(
        self,
        db: Session,
        job_id: int,
        error_message: str,
    ) -> ProcessingJob:
        """
        将运行中的任务标记为失败。

        error_message必须是调用方提供的、
        已脱敏业务错误摘要。
        """

        job = self._get_job(
            db=db,
            job_id=job_id,
        )

        StatusMachine.transition_processing_job(
            job=job,
            target_status=(
                ProcessingJobStatus.FAILED
            ),
        )

        normalized_message = (
            error_message.strip()
        )

        if not normalized_message:
            normalized_message = (
                "processing job failed"
            )

        job.error_message = normalized_message[
            :self.MAX_ERROR_MESSAGE_LENGTH
        ]

        job.finished_at = datetime.utcnow()

        db.commit()
        db.refresh(job)

        return job

    def get_job(
        self,
        db: Session,
        job_id: int,
    ) -> ProcessingJob:
        """
        查询指定任务。
        """

        return self._get_job(
            db=db,
            job_id=job_id,
        )

    def get_latest_document_job(
        self,
        db: Session,
        document_id: int,
    ) -> ProcessingJob | None:
        """
        查询文档最近一次任务。
        """

        return (
            self.processing_job_repository
            .find_latest_by_document_id(
                db=db,
                document_id=document_id,
            )
        )

    def _get_job(
        self,
        db: Session,
        job_id: int,
    ) -> ProcessingJob:
        """
        查询任务，不存在时抛出业务异常。
        """

        job = (
            self.processing_job_repository
            .find_by_id(
                db=db,
                job_id=job_id,
            )
        )

        if job is None:
            raise ProcessingJobNotFoundError(
                "processing job not found"
            )

        return job

    @staticmethod
    def _validate_document_status(
        document_status: DocumentStatus,
        job_type: ProcessingJobType,
    ) -> None:
        """
        校验文档状态是否允许创建任务。
        """

        allowed_statuses = {
            ProcessingJobType.DOCUMENT_PROCESSING: {
                DocumentStatus.UPLOADED,
                DocumentStatus.PARSE_FAILED,
                DocumentStatus.PARSED,
                DocumentStatus.CHUNK_FAILED,
            },
            ProcessingJobType.EMBEDDING: {
                DocumentStatus.CHUNKED,
                DocumentStatus.EMBEDDING_FAILED,
            },
        }

        if document_status not in allowed_statuses[
            job_type
        ]:
            raise InvalidProcessingJobError(
                "document status does not allow "
                f"{job_type.value} job: "
                f"{document_status.value}"
            )