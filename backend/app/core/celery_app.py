from celery import Celery

from app.core.config import get_settings
from app.core.logging_config import configure_application_logging


settings = get_settings()
configure_application_logging(settings.log_level)

celery_app = Celery(
    "knowledge_assistant",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=[
        "app.tasks.infrastructure",
        "app.tasks.processing_job",
    ],
)

visibility_timeout = settings.celery_visibility_timeout
celery_app.conf.update(
    accept_content=["json"],
    task_serializer="json",
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_ignore_result=True,
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,
    broker_connection_retry_on_startup=True,
    broker_transport_options={"visibility_timeout": visibility_timeout},
    result_backend_transport_options={"visibility_timeout": visibility_timeout},
    visibility_timeout=visibility_timeout,
    task_always_eager=settings.celery_task_always_eager,
    task_eager_propagates=settings.celery_task_eager_propagates,
)
