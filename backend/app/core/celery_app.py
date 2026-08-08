from celery import Celery

from app.core.config import get_settings


settings = get_settings()
celery_app = Celery(
    "knowledge_assistant",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=["app.tasks.infrastructure"],
)

celery_app.conf.update(
    accept_content=["json"],
    task_serializer="json",
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_ignore_result=True,
    broker_connection_retry_on_startup=True,
    task_always_eager=settings.celery_task_always_eager,
    task_eager_propagates=settings.celery_task_eager_propagates,
)
