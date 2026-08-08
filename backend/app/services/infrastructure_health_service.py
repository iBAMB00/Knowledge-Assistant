from collections.abc import Callable
from typing import Any

from sqlalchemy import text
from sqlalchemy.orm import Session


def create_redis_client(redis_url: str, **kwargs):
    """延迟创建 Redis Client，避免普通 API 进程启动时提前建立连接。"""
    from redis import Redis

    return Redis.from_url(redis_url, **kwargs)


class InfrastructureHealthService:
    """检查 API 依赖的数据库与 Redis 是否就绪。"""

    def __init__(
        self,
        redis_url: str,
        redis_factory: Callable[..., Any] = create_redis_client,
    ) -> None:
        self.redis_url = redis_url
        self.redis_factory = redis_factory

    def check_readiness(self, db: Session) -> dict[str, str]:
        """执行轻量数据库查询与 Redis PING。"""
        db.execute(text("SELECT 1"))
        redis_client = self.redis_factory(
            self.redis_url,
            socket_connect_timeout=1,
            socket_timeout=1,
            decode_responses=True,
        )

        try:
            redis_client.ping()
        finally:
            close = getattr(redis_client, "close", None)
            if callable(close):
                close()

        return {"status": "ready", "database": "ok", "redis": "ok"}
