from app.core.celery_app import celery_app


@celery_app.task(name="infrastructure.ping", ignore_result=False)
def infrastructure_ping() -> dict[str, str]:
    """用于验证 Celery Worker 与 Redis Broker 是否可用。"""
    return {"status": "ok"}
