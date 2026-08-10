from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.security import decode_access_token
from app.models.database.user import User
from app.repositories.user_repository import UserRepository
from app.services.auth_service import AuthService, AuthenticationError


bearer_scheme = HTTPBearer(auto_error=False)
auth_service = AuthService(UserRepository())


def _unauthorized() -> HTTPException:
    """构造统一 Bearer 认证失败响应。"""
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="invalid or missing access token",
        headers={"WWW-Authenticate": "Bearer"},
    )


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(
        bearer_scheme
    ),
    db: Session = Depends(get_db),
) -> User:
    """解析 Bearer JWT 并返回当前 active 用户。"""
    if (
        credentials is None
        or credentials.scheme.lower() != "bearer"
    ):
        raise _unauthorized()

    try:
        user_id = decode_access_token(credentials.credentials)
        return auth_service.get_active_user(
            db=db,
            user_id=user_id,
        )
    except (ValueError, AuthenticationError) as exc:
        raise _unauthorized() from exc
