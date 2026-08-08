from app.core.celery_app import celery_app
from app.services.processing_job_runtime import get_processing_job_runner


@celery_app.task(name="processing_job.execute", ignore_result=True)
def execute_processing_job(job_id: int) -> None:
    """由 Celery Worker 根据 job_id 执行持久化任务。"""
    get_processing_job_runner().run(job_id)
