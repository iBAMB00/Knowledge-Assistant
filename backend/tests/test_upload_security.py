import asyncio

import pytest
from pydantic import ValidationError

from app.api.upload_security import read_upload_with_limit
from app.core.config import Settings
from app.services.document_upload_policy import (
    DocumentUploadMediaTypeError,
    DocumentUploadPolicy,
    DocumentUploadTooLargeError,
    DocumentUploadValidationError,
)


def build_settings(**overrides) -> Settings:
    """构造不依赖真实外部服务的 Settings。"""
    values = {
        "model_provider": "test",
        "model_base_url": "http://model.test",
        "model_name": "test-model",
        "model_api_key": "test-model-key",
        "embedding_provider": "test",
        "embedding_base_url": "http://embedding.test",
        "embedding_model": "test-embedding",
        "embedding_api_key": "test-embedding-key",
        "jwt_secret_key": "a" * 48,
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


def test_document_upload_policy_accepts_supported_documents() -> None:
    """验证支持格式同时通过扩展名、MIME 和基础内容校验。"""
    policy = DocumentUploadPolicy(max_file_size_bytes=1024)

    assert policy.validate(
        filename=r"C:\fakepath\guide.md",
        content="# Guide\n正文".encode("utf-8"),
        content_type="text/markdown; charset=utf-8",
    ) == "guide.md"

    assert policy.validate(
        filename="manual.pdf",
        content=b"%PDF-1.7\nmock",
        content_type="application/pdf",
    ) == "manual.pdf"


def test_document_upload_policy_rejects_oversized_or_mismatched_content() -> None:
    """验证大小限制、伪装扩展名和 MIME 不匹配均被拒绝。"""
    policy = DocumentUploadPolicy(max_file_size_bytes=16)

    with pytest.raises(DocumentUploadTooLargeError):
        policy.validate(
            filename="large.txt",
            content=b"12345678901234567",
            content_type="text/plain",
        )

    with pytest.raises(DocumentUploadMediaTypeError):
        policy.validate(
            filename="fake.pdf",
            content=b"not-a-pdf",
            content_type="application/pdf",
        )

    with pytest.raises(DocumentUploadMediaTypeError):
        policy.validate(
            filename="guide.txt",
            content=b"hello",
            content_type="image/png",
        )

    with pytest.raises(DocumentUploadMediaTypeError):
        policy.validate(
            filename="payload.exe",
            content=b"hello",
            content_type="application/octet-stream",
        )


def test_document_upload_policy_rejects_unsafe_filename() -> None:
    """验证危险或平台保留文件名不会进入存储层。"""
    policy = DocumentUploadPolicy(max_filename_length=32)

    with pytest.raises(DocumentUploadValidationError):
        policy.validate(
            filename="CON.txt",
            content=b"hello",
            content_type="text/plain",
        )

    with pytest.raises(DocumentUploadValidationError):
        policy.validate(
            filename="bad\x00name.txt",
            content=b"hello",
            content_type="text/plain",
        )

    with pytest.raises(DocumentUploadValidationError):
        policy.validate(
            filename=f"{'a' * 40}.txt",
            content=b"hello",
            content_type="text/plain",
        )


def test_production_settings_reject_unsafe_runtime_configuration() -> None:
    """验证生产模式拒绝 Debug、CORS wildcard 与示例弱密钥。"""
    with pytest.raises(ValidationError):
        build_settings(
            app_environment="production",
            debug=True,
        )

    with pytest.raises(ValidationError):
        build_settings(
            app_environment="production",
            cors_allowed_origins=["*"],
        )

    with pytest.raises(ValidationError):
        build_settings(
            app_environment="production",
            jwt_secret_key="replace_with_a_random_secret_at_least_32_chars",
        )

    with pytest.raises(ValidationError):
        build_settings(
            app_environment="production",
            storage_backend="minio",
            minio_access_key="knowledge_assistant",
            minio_secret_key="knowledge_assistant_dev_secret",
        )


def test_cors_credentials_cannot_use_wildcard_in_any_environment() -> None:
    """验证携带凭证的 CORS 不允许 wildcard。"""
    with pytest.raises(ValidationError):
        build_settings(
            cors_allowed_origins=["*"],
            cors_allow_credentials=True,
        )


def test_valid_production_settings_are_accepted() -> None:
    """验证显式生产安全配置可以正常启动。"""
    settings = build_settings(
        app_environment="production",
        cors_allowed_origins=["https://knowledge.example.com"],
        jwt_secret_key="prod-secret-" + "x" * 40,
    )

    assert settings.app_environment == "production"
    assert settings.debug is False


class FakeUploadFile:
    """记录 Router 边界读取大小的最小 UploadFile 替身。"""

    def __init__(self, content: bytes) -> None:
        self.content = content
        self.requested_size: int | None = None

    async def read(self, size: int = -1) -> bytes:
        self.requested_size = size
        return self.content[:size]


def test_read_upload_with_limit_reads_only_limit_plus_one() -> None:
    """验证 Router 不会先把超大文件完整读入内存。"""
    policy = DocumentUploadPolicy(max_file_size_bytes=8)
    file = FakeUploadFile(content=b"x" * 100)

    async def run() -> None:
        with pytest.raises(DocumentUploadTooLargeError):
            await read_upload_with_limit(
                file,  # type: ignore[arg-type]
                policy,
            )

    asyncio.run(run())

    assert file.requested_size == 9
