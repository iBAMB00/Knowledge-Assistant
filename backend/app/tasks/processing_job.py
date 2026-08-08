import logging

from app.core.celery_app import celery_app
from app.core.config import get_settings
from app.services.processing_job_retry_policy import ProcessingJobRetryPolicy
from app.services.processing_job_runtime import get_processing_job_runner
from app.services.processing_job_service import ProcessingJobAlreadyClaimedError

logger = logging.getLogger(__name__)
settings = get_settings()
retry_policy = ProcessingJobRetryPolicy(
    base_delay_seconds=settings.processing_job_retry_base_delay,
    max_delay_seconds=settings.processing_job_retry_max_delay,
)


@celery_app.task(
    bind=True,
    name="processing_job.execute",
    ignore_result=True,
    acks_late=True,
    reject_on_worker_lost=True,
    max_retries=settings.processing_job_max_retries,
)
def execute_processing_job(self, job_id: int) -> None:
    """领取并执行持久化任务；瞬时故障重试，终态重复消息直接跳过。"""
    runner = get_processing_job_runner()

    delivery_info = getattr(self.request, "delivery_info", {}) or {}
    force_resume = bool(delivery_info.get("redelivered"))

    try:
        runner.run_worker(job_id, force_resume=force_resume)
        return
    except ProcessingJobAlreadyClaimedError as exc:
        logger.info(
            "processing job already owned by another worker, skip duplicate: "
            "job_id=%s, lease_remaining=%ss",
            job_id,
            exc.retry_after_seconds,
        )
        return
    except Exception as exc:
        if (
            retry_policy.should_retry(exc)
            and self.request.retries < settings.processing_job_max_retries
        ):
            try:
                runner.release_for_retry(job_id)
            except Exception:
                logger.warning(
                    "processing job retry lease release failed: job_id=%s",
                    job_id,
                    exc_info=True,
                )

            countdown = retry_policy.retry_delay_seconds(self.request.retries)
            logger.warning(
                "processing job transient failure, retry scheduled: "
                "job_id=%s, retry=%s/%s, countdown=%ss, error_type=%s",
                job_id,
                self.request.retries + 1,
                settings.processing_job_max_retries,
                countdown,
                type(exc).__name__,
            )
            raise self.retry(exc=exc, countdown=countdown) from exc

        try:
            runner.fail(job_id)
        except Exception:
            logger.error(
                "processing job final failure state could not be persisted: job_id=%s",
                job_id,
                exc_info=True,
            )
        raise
