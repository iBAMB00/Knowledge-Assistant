import logging
from collections.abc import Callable

from sqlalchemy.orm import Session

from app.services.processing_job_executor import ProcessingJobExecutor
from app.services.processing_job_recovery_service import ProcessingJobRecoveryService
from app.services.processing_job_service import (
    ProcessingJobAlreadyClaimedError,
    ProcessingJobService,
)

logger = logging.getLogger(__name__)


class ProcessingJobRunner:
    """为同步兼容入口和 Celery Worker 提供独立数据库 Session。"""

    def __init__(
        self,
        session_factory: Callable[[], Session],
        executor: ProcessingJobExecutor,
        processing_job_service: ProcessingJobService | None = None,
        recovery_service: ProcessingJobRecoveryService | None = None,
    ) -> None:
        self.session_factory = session_factory
        self.executor = executor
        self.processing_job_service = processing_job_service or getattr(
            executor, "processing_job_service", None
        )
        self.recovery_service = recovery_service

    def run(self, job_id: int) -> None:
        """同步兼容：创建独立 Session 并执行 pending 任务。"""
        db = self.session_factory()
        try:
            self.executor.execute_job(db=db, job_id=job_id)
        except Exception:
            db.rollback()
            logger.exception("processing job runner failed: job_id=%s", job_id)
            raise
        finally:
            db.close()

    def run_worker(self, job_id: int, force_resume: bool = False) -> bool:
        """Worker 领取并执行任务；终态重复消息直接返回 False。"""
        db = self.session_factory()
        try:
            processing_job_service = self._require_processing_job_service()
            claim = processing_job_service.claim_job_for_worker(
                db=db,
                job_id=job_id,
                force_resume=force_resume,
            )
            if claim is None:
                logger.info("processing job already terminal, skip duplicate: job_id=%s", job_id)
                return False

            if claim.resumed:
                if self.recovery_service is None:
                    raise RuntimeError("processing job recovery service is required for resume")
                job = processing_job_service.get_job(db=db, job_id=job_id)
                self.recovery_service.prepare_for_resume(db=db, job=job)

            self.executor.execute_claimed_job(db=db, job_id=job_id)
            return True
        except ProcessingJobAlreadyClaimedError:
            db.rollback()
            raise
        except Exception:
            db.rollback()
            logger.exception("processing job worker execution failed: job_id=%s", job_id)
            raise
        finally:
            db.close()

    def release_for_retry(self, job_id: int) -> None:
        """瞬时失败后释放租约，允许 Celery retry 重新领取。"""
        db = self.session_factory()
        try:
            self._require_processing_job_service().release_job_for_retry(
                db=db, job_id=job_id
            )
        except Exception:
            db.rollback()
            logger.exception("failed to release processing job lease: job_id=%s", job_id)
            raise
        finally:
            db.close()

    def fail(self, job_id: int) -> None:
        """重试耗尽或永久异常时把业务任务标为 failed。"""
        db = self.session_factory()
        try:
            self._require_processing_job_service().fail_job_for_execution_error(
                db=db, job_id=job_id
            )
        except Exception:
            db.rollback()
            logger.exception("failed to finalize processing job failure: job_id=%s", job_id)
            raise
        finally:
            db.close()

    def _require_processing_job_service(self) -> ProcessingJobService:
        """Worker 可靠性入口必须显式拥有 ProcessingJobService。"""
        if self.processing_job_service is None:
            raise RuntimeError("processing job service is required for worker execution")
        return self.processing_job_service
