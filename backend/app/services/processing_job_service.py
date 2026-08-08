from datetime import datetime

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.constants.document_status import DocumentStatus
from app.constants.processing_job_stage import ProcessingJobStage
from app.constants.processing_job_status import ProcessingJobStatus
from app.constants.processing_job_type import ProcessingJobType
from app.models.database.processing_job import ProcessingJob
from app.repositories.document_repository import DocumentRepository
from app.repositories.processing_job_repository import (
    ProcessingJobRepository,
)
from app.services.status_machine import StatusMachine


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
    - 管理任务状态、阶段和进度
    - 管理事务边界

    不负责真正执行文档处理任务。
    """

    MAX_ERROR_MESSAGE_LENGTH = 500

    def __init__(
        self,
        document_repository: DocumentRepository,
        processing_job_repository: ProcessingJobRepository,
    ) -> None:
        self.document_repository = document_repository
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

        document = self.document_repository.find_by_id(
            db=db,
            document_id=document_id,
        )

        if document is None:
            raise ValueError("document not found")

        self._validate_document_status(
            document_status=DocumentStatus(document.status),
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
            status=ProcessingJobStatus.PENDING.value,
            stage=ProcessingJobStage.QUEUED.value,
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

            if self._is_active_job_conflict(exc):
                raise ActiveProcessingJobError(
                    "document already has an active job"
                ) from exc

            raise

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

        任务刚开始运行时仍处于queued阶段，
        由Executor在进入真实业务步骤时更新stage。
        """

        job = self._get_job(db=db, job_id=job_id)

        StatusMachine.transition_processing_job(
            job=job,
            target_status=ProcessingJobStatus.RUNNING,
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

        保留该方法用于兼容已有调用；
        新阶段编排优先使用update_stage。
        """

        self._validate_running_progress(progress)

        job = self._get_job(db=db, job_id=job_id)
        self._ensure_running(job)
        self._ensure_progress_not_decreasing(
            current_progress=job.progress,
            target_progress=progress,
        )

        job.progress = progress

        db.commit()
        db.refresh(job)
        return job

    def update_stage(
        self,
        db: Session,
        job_id: int,
        stage: ProcessingJobStage,
        progress: int,
    ) -> ProcessingJob:
        """
        同步更新运行中任务的业务阶段和整体进度。

        stage由状态机保证只能向前推进；
        progress必须处于0到99且不能倒退。
        """

        self._validate_running_progress(progress)

        job = self._get_job(db=db, job_id=job_id)
        self._ensure_running(job)
        self._ensure_progress_not_decreasing(
            current_progress=job.progress,
            target_progress=progress,
        )

        StatusMachine.transition_processing_job_stage(
            job=job,
            target_stage=stage,
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
        将finalizing阶段的运行中任务标记为成功。
        """

        job = self._get_job(db=db, job_id=job_id)
        self._ensure_running(job)

        if ProcessingJobStage(job.stage) != ProcessingJobStage.FINALIZING:
            raise InvalidProcessingJobError(
                "job must be finalizing before success"
            )

        StatusMachine.transition_processing_job_stage(
            job=job,
            target_stage=ProcessingJobStage.COMPLETED,
        )
        StatusMachine.transition_processing_job(
            job=job,
            target_status=ProcessingJobStatus.SUCCEEDED,
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
        将 pending 或 running 任务标记为失败。

        pending 可用于记录任务派发失败；
        running 失败时保留最后stage和progress；
        error_message必须是调用方提供的已脱敏业务摘要。
        """

        job = self._get_job(db=db, job_id=job_id)

        StatusMachine.transition_processing_job(
            job=job,
            target_status=ProcessingJobStatus.FAILED,
        )

        normalized_message = error_message.strip()

        if not normalized_message:
            normalized_message = "processing job failed"

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

        return self._get_job(db=db, job_id=job_id)

    def get_latest_document_job(
        self,
        db: Session,
        document_id: int,
    ) -> ProcessingJob:
        """
        查询文档最近一次处理任务。
        """

        self._ensure_document_exists(
            db=db,
            document_id=document_id,
        )

        job = (
            self.processing_job_repository
            .find_latest_by_document_id(
                db=db,
                document_id=document_id,
            )
        )

        if job is None:
            raise ProcessingJobNotFoundError(
                "processing job not found"
            )

        return job

    def list_document_jobs(
        self,
        db: Session,
        document_id: int,
    ) -> list[ProcessingJob]:
        """
        查询文档全部处理任务。

        文档不存在时抛出业务异常；
        文档存在但没有任务时返回空列表。
        """

        self._ensure_document_exists(
            db=db,
            document_id=document_id,
        )

        return (
            self.processing_job_repository
            .find_all_by_document_id(
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

        job = self.processing_job_repository.find_by_id(
            db=db,
            job_id=job_id,
        )

        if job is None:
            raise ProcessingJobNotFoundError(
                "processing job not found"
            )

        return job

    def _ensure_document_exists(
        self,
        db: Session,
        document_id: int,
    ) -> None:
        """确认文档存在。"""

        document = self.document_repository.find_by_id(
            db=db,
            document_id=document_id,
        )

        if document is None:
            raise ValueError("document not found")

    @staticmethod
    def _ensure_running(job: ProcessingJob) -> None:
        """确认任务处于running状态。"""

        if (
            ProcessingJobStatus(job.status)
            != ProcessingJobStatus.RUNNING
        ):
            raise InvalidProcessingJobError(
                "only running job can update stage or progress"
            )

    @staticmethod
    def _validate_running_progress(progress: int) -> None:
        """校验运行中任务的进度范围。"""

        if progress < 0 or progress >= 100:
            raise ValueError(
                "running job progress must be between 0 and 99"
            )

    @staticmethod
    def _ensure_progress_not_decreasing(
        current_progress: int,
        target_progress: int,
    ) -> None:
        """阻止任务进度倒退。"""

        if target_progress < current_progress:
            raise InvalidProcessingJobError(
                "processing job progress cannot decrease"
            )

    def _validate_document_status(
        self,
        document_status: DocumentStatus,
        job_type: ProcessingJobType,
    ) -> None:
        """校验文档状态是否允许创建指定任务。"""

        if job_type == ProcessingJobType.DOCUMENT_PROCESSING:
            allowed_statuses = {
                DocumentStatus.UPLOADED,
                DocumentStatus.PARSE_FAILED,
                DocumentStatus.PARSED,
                DocumentStatus.CHUNK_FAILED,
            }

            if document_status not in allowed_statuses:
                raise InvalidProcessingJobError(
                    f"document status {document_status.value} "
                    "does not allow document processing job"
                )

            return

        if job_type == ProcessingJobType.EMBEDDING:
            allowed_statuses = {
                DocumentStatus.CHUNKED,
                DocumentStatus.EMBEDDING_FAILED,
                DocumentStatus.COMPLETED,
            }

            if document_status not in allowed_statuses:
                raise InvalidProcessingJobError(
                    f"document status {document_status.value} "
                    "does not allow embedding job"
                )

            return

        if job_type == ProcessingJobType.FULL_PIPELINE:
            allowed_statuses = {
                DocumentStatus.UPLOADED,
                DocumentStatus.PARSE_FAILED,
                DocumentStatus.PARSED,
                DocumentStatus.CHUNK_FAILED,
                DocumentStatus.CHUNKED,
                DocumentStatus.EMBEDDING_FAILED,
            }

            if document_status not in allowed_statuses:
                raise InvalidProcessingJobError(
                    f"document status {document_status.value} "
                    "does not allow full pipeline job"
                )

            return

        raise InvalidProcessingJobError(
            f"unsupported processing job type: {job_type.value}"
        )

    @staticmethod
    def _is_active_job_conflict(
        exc: IntegrityError,
    ) -> bool:
        """
        判断完整性错误是否由活动任务唯一约束引起。

        当前兼容SQLite，同时为未来PostgreSQL保留约束名判断。
        """

        original_error = exc.orig
        error_message = str(original_error).lower()

        if (
            "unique constraint failed" in error_message
            and "processing_jobs.document_id" in error_message
        ):
            return True

        diagnostic = getattr(original_error, "diag", None)
        constraint_name = getattr(
            diagnostic,
            "constraint_name",
            None,
        )

        return constraint_name in {
            "uq_processing_jobs_active_document",
            "uq_processing_jobs_active_document_id",
        }
