from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.database import get_db
from app.services.infrastructure_health_service import InfrastructureHealthService


router = APIRouter(tags=["Health"])
settings = get_settings()
health_service = InfrastructureHealthService(redis_url=settings.redis_url)


@router.get("/health")
def health() -> dict[str, str]:
    """进程存活检查，不访问外部依赖。"""
    return {"status": "ok"}


@router.get("/health/ready")
def readiness(db: Session = Depends(get_db)) -> dict[str, str]:
    """检查数据库与 Redis 是否可用。"""
    try:
        return health_service.check_readiness(db)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="infrastructure not ready",
        ) from exc
