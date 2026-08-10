from datetime import datetime, timedelta, timezone

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError

from app.core.config import get_settings


settings = get_settings()
password_hasher = PasswordHasher()


def hash_password(password: str) -> str:
    """使用 Argon2 对明文密码进行不可逆哈希。"""
    return password_hasher.hash(password)


def verify_password(
    password: str,
    password_hash: str,
) -> bool:
    """校验明文密码是否匹配已保存的 Argon2 哈希。"""
    try:
        return password_hasher.verify(
            password_hash,
            password,
        )
    except (VerificationError, InvalidHashError):
        return False


def create_access_token(user_id: int) -> tuple[str, int]:
    """为指定用户签发短期访问 Token。"""
    now = datetime.now(timezone.utc)
    expires_in = settings.jwt_access_token_expire_minutes * 60
    expires_at = now + timedelta(seconds=expires_in)

    payload = {
        "sub": str(user_id),
        "type": "access",
        "iat": now,
        "exp": expires_at,
    }

    token = jwt.encode(
        payload,
        settings.jwt_secret_key,
        algorithm=settings.jwt_algorithm,
    )

    return token, expires_in


def decode_access_token(token: str) -> int:
    """
    校验访问 Token 并返回用户 ID。

    任何签名、过期、类型或 subject 异常都统一抛出 ValueError，
    避免 API 层依赖具体 JWT 库异常。
    """
    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret_key,
            algorithms=[settings.jwt_algorithm],
            options={
                "require": [
                    "sub",
                    "type",
                    "iat",
                    "exp",
                ]
            },
        )
    except jwt.InvalidTokenError as exc:
        raise ValueError("invalid access token") from exc

    if payload.get("type") != "access":
        raise ValueError("invalid access token")

    subject = payload.get("sub")

    try:
        user_id = int(subject)
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid access token") from exc

    if user_id <= 0:
        raise ValueError("invalid access token")

    return user_id
