from sqlalchemy import text

from app.core.database import build_engine_options
from app.services.infrastructure_health_service import InfrastructureHealthService


class FakeRedis:
    def __init__(self) -> None:
        self.ping_called = False
        self.closed = False

    def ping(self) -> bool:
        self.ping_called = True
        return True

    def close(self) -> None:
        self.closed = True


def test_build_engine_options_keeps_sqlite_specific_connect_args() -> None:
    options = build_engine_options("sqlite:///./test.db")
    assert options["connect_args"] == {"check_same_thread": False}
    assert "pool_recycle" not in options


def test_build_engine_options_supports_postgresql_without_sqlite_args() -> None:
    options = build_engine_options("postgresql+psycopg://user:pass@localhost/db")
    assert "connect_args" not in options
    assert options["pool_recycle"] > 0
    assert options["pool_pre_ping"] is True


def test_infrastructure_health_checks_database_and_redis(db) -> None:
    fake_redis = FakeRedis()

    def fake_redis_factory(*args, **kwargs):
        return fake_redis

    service = InfrastructureHealthService(
        redis_url="redis://localhost:6379/0",
        redis_factory=fake_redis_factory,
    )

    result = service.check_readiness(db)
    assert db.execute(text("SELECT 1")).scalar_one() == 1
    assert fake_redis.ping_called is True
    assert fake_redis.closed is True
    assert result == {"status": "ready", "database": "ok", "redis": "ok"}
