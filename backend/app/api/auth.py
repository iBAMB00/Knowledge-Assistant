from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.dependencies.auth import get_current_user
from app.core.database import get_db
from app.models.database.user import User
from app.repositories.user_repository import UserRepository
from app.schemas.token_response import TokenResponse
from app.schemas.user_login_request import UserLoginRequest
from app.schemas.user_register_request import UserRegisterRequest
from app.schemas.user_response import UserResponse
from app.services.auth_service import AuthService, AuthenticationError


router = APIRouter(
    prefix="/auth",
    tags=["Authentication"],
)

auth_service = AuthService(UserRepository())


@router.post(
    "/register",
    status_code=status.HTTP_201_CREATED,
    response_model=UserResponse,
)
def register_user(
    request: UserRegisterRequest,
    db: Session = Depends(get_db),
) -> User:
    """创建一个新的本地认证用户。"""
    try:
        return auth_service.register(
            db=db,
            email=str(request.email),
            password=request.password,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="用户注册失败",
        ) from exc


@router.post(
    "/login",
    response_model=TokenResponse,
)
def login_user(
    request: UserLoginRequest,
    db: Session = Depends(get_db),
) -> TokenResponse:
    """校验用户凭据并返回 Bearer access token。"""
    try:
        return auth_service.login(
            db=db,
            email=str(request.email),
            password=request.password,
        )
    except AuthenticationError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc


@router.get(
    "/me",
    response_model=UserResponse,
)
def get_me(
    current_user: User = Depends(get_current_user),
) -> User:
    """返回当前登录用户。"""
    return current_user
