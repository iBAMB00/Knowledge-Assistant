from datetime import datetime, timedelta, timezone

import jwt
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.auth import router as auth_router
from app.core.config import get_settings
from app.core.database import get_db
from app.core.security import hash_password, verify_password
from app.models.database.user import User


settings = get_settings()


def _build_client(db):
    """创建只挂载 Auth Router 的测试应用，避免依赖外部模型服务。"""
    app = FastAPI()
    app.include_router(auth_router)

    def override_get_db():
        yield db

    app.dependency_overrides[get_db] = override_get_db
    return TestClient(app)


def test_password_is_hashed_with_argon2():
    password_hash = hash_password("StrongPass123")

    assert password_hash != "StrongPass123"
    assert password_hash.startswith("$argon2")
    assert verify_password("StrongPass123", password_hash) is True
    assert verify_password("wrong-password", password_hash) is False


def test_register_login_and_me(db):
    client = _build_client(db)

    register_response = client.post(
        "/auth/register",
        json={
            "email": "User@Example.com",
            "password": "StrongPass123",
        },
    )

    assert register_response.status_code == 201
    user_body = register_response.json()
    assert user_body["email"] == "user@example.com"
    assert user_body["is_active"] is True
    assert "password_hash" not in user_body

    stored_user = (
        db.query(User)
        .filter(User.email == "user@example.com")
        .one()
    )
    assert stored_user.password_hash != "StrongPass123"

    login_response = client.post(
        "/auth/login",
        json={
            "email": "USER@example.com",
            "password": "StrongPass123",
        },
    )

    assert login_response.status_code == 200
    token_body = login_response.json()
    assert token_body["token_type"] == "bearer"
    assert token_body["expires_in"] == 3600
    assert token_body["access_token"]

    me_response = client.get(
        "/auth/me",
        headers={
            "Authorization": (
                f"Bearer {token_body['access_token']}"
            )
        },
    )

    assert me_response.status_code == 200
    assert me_response.json()["id"] == stored_user.id
    assert me_response.json()["email"] == "user@example.com"


def test_duplicate_email_returns_409(db):
    client = _build_client(db)
    payload = {
        "email": "duplicate@example.com",
        "password": "StrongPass123",
    }

    first = client.post("/auth/register", json=payload)
    second = client.post(
        "/auth/register",
        json={**payload, "email": "DUPLICATE@EXAMPLE.COM"},
    )

    assert first.status_code == 201
    assert second.status_code == 409
    assert second.json()["detail"] == "email already registered"


def test_invalid_credentials_return_401(db):
    client = _build_client(db)
    client.post(
        "/auth/register",
        json={
            "email": "login@example.com",
            "password": "StrongPass123",
        },
    )

    response = client.post(
        "/auth/login",
        json={
            "email": "login@example.com",
            "password": "WrongPass123",
        },
    )

    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"


def test_me_requires_valid_bearer_token(db):
    client = _build_client(db)

    missing = client.get("/auth/me")
    invalid = client.get(
        "/auth/me",
        headers={"Authorization": "Bearer not-a-token"},
    )

    assert missing.status_code == 401
    assert invalid.status_code == 401
    assert missing.headers["www-authenticate"] == "Bearer"


def test_inactive_user_is_rejected_even_with_valid_token(db):
    client = _build_client(db)

    client.post(
        "/auth/register",
        json={
            "email": "inactive@example.com",
            "password": "StrongPass123",
        },
    )
    login = client.post(
        "/auth/login",
        json={
            "email": "inactive@example.com",
            "password": "StrongPass123",
        },
    )
    token = login.json()["access_token"]

    user = (
        db.query(User)
        .filter(User.email == "inactive@example.com")
        .one()
    )
    user.is_active = False
    db.commit()

    response = client.get(
        "/auth/me",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 401


def test_expired_token_is_rejected(db):
    client = _build_client(db)

    user = User(
        email="expired@example.com",
        password_hash=hash_password("StrongPass123"),
        is_active=True,
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    now = datetime.now(timezone.utc)
    token = jwt.encode(
        {
            "sub": str(user.id),
            "type": "access",
            "iat": now - timedelta(minutes=10),
            "exp": now - timedelta(minutes=1),
        },
        settings.jwt_secret_key,
        algorithm=settings.jwt_algorithm,
    )

    response = client.get(
        "/auth/me",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 401


def test_token_missing_exp_is_rejected(db):
    client = _build_client(db)

    user = User(
        email="missing-exp@example.com",
        password_hash=hash_password("StrongPass123"),
        is_active=True,
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    now = datetime.now(timezone.utc)
    token = jwt.encode(
        {
            "sub": str(user.id),
            "type": "access",
            "iat": now,
        },
        settings.jwt_secret_key,
        algorithm=settings.jwt_algorithm,
    )

    response = client.get(
        "/auth/me",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 401
