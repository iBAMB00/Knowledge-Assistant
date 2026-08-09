from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.security import (
    create_access_token,
    hash_password,
    verify_password,
)
from app.models.database.user import User
from app.repositories.user_repository import UserRepository
from app.schemas.token_response import TokenResponse


class AuthenticationError(ValueError):
    """认证凭据无效或用户不可用。"""


class AuthService:
    """用户注册、登录与当前用户解析业务服务。"""

    def __init__(
        self,
        user_repository: UserRepository,
    ) -> None:
        self.user_repository = user_repository

    @staticmethod
    def normalize_email(email: str) -> str:
        """统一邮箱大小写和首尾空白。"""
        return email.strip().lower()

    def register(
        self,
        db: Session,
        email: str,
        password: str,
    ) -> User:
        """注册新用户，由 Service 负责事务提交与回滚。"""
        normalized_email = self.normalize_email(email)

        if self.user_repository.find_by_email(
            db=db,
            email=normalized_email,
        ):
            raise ValueError("email already registered")

        user = User(
            email=normalized_email,
            password_hash=hash_password(password),
            is_active=True,
        )

        try:
            self.user_repository.create(
                db=db,
                user=user,
            )
            db.commit()
            db.refresh(user)
            return user
        except IntegrityError as exc:
            db.rollback()
            raise ValueError("email already registered") from exc
        except Exception:
            db.rollback()
            raise

    def login(
        self,
        db: Session,
        email: str,
        password: str,
    ) -> TokenResponse:
        """校验邮箱密码并签发访问 Token。"""
        user = self.user_repository.find_by_email(
            db=db,
            email=self.normalize_email(email),
        )

        if (
            user is None
            or not verify_password(
                password,
                user.password_hash,
            )
        ):
            raise AuthenticationError(
                "invalid email or password"
            )

        if not user.is_active:
            raise AuthenticationError("user is inactive")

        access_token, expires_in = create_access_token(user.id)

        return TokenResponse(
            access_token=access_token,
            expires_in=expires_in,
        )

    def get_active_user(
        self,
        db: Session,
        user_id: int,
    ) -> User:
        """读取仍处于 active 状态的用户。"""
        user = self.user_repository.find_by_id(
            db=db,
            user_id=user_id,
        )

        if user is None or not user.is_active:
            raise AuthenticationError("invalid access token")

        return user
