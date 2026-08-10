from functools import lru_cache
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """
    应用配置。

    管理：
    - LLM模型配置
    - Embedding模型配置
    - 数据库配置
    """

    app_name: str = (
        "Knowledge Assistant"
    )

    debug: bool = False

    app_environment: Literal[
        "development",
        "test",
        "production",
    ] = "development"

    cors_allowed_origins: list[str] = Field(default_factory=list)
    cors_allow_credentials: bool = False

    upload_max_file_size_bytes: int = Field(
        default=20 * 1024 * 1024,
        gt=0,
    )
    upload_max_filename_length: int = Field(
        default=255,
        gt=0,
        le=1024,
    )

    log_level: Literal[
        "DEBUG",
        "INFO",
        "WARNING",
        "ERROR",
        "CRITICAL",
    ] = "INFO"


    # ==========================
    # LLM配置
    # ==========================

    model_provider: str

    model_base_url: str

    model_name: str

    model_api_key: str


    # ==========================
    # Embedding配置
    # ==========================

    embedding_provider: str

    embedding_base_url: str

    embedding_model: str

    embedding_api_key: str

    embedding_dimension: int = 1024

    # ==========================
    # Chunk配置
    # ==========================

    chunk_strategy: str = (
        "recursive_character"
    )

    chunk_size: int = 600

    chunk_overlap: int = 100

    structure_aware_parent_enabled: bool = True

    parent_child_enabled: bool = True

    parent_child_child_size: int = Field(
        default=300,
        gt=0,
    )

    parent_child_child_overlap: int = Field(
        default=50,
        ge=0,
    )

    # ==========================
    # Retrieval配置
    # ==========================

    retrieval_top_k: int = Field(
        default=5,
        gt=0,
    )

    retrieval_candidate_k: int = Field(
        default=20,
        gt=0,
    )

    retrieval_score_threshold: float = Field(
        default=-1.0,
        ge=-1.0,
        le=1.0,
    )

    knowledge_chat_score_threshold: float = Field(
        default=0.40,
        ge=-1.0,
        le=1.0,
    )

    retrieval_per_document_limit: int = Field(
        default=2,
        gt=0,
    )

    retrieval_hybrid_enabled: bool = True

    retrieval_rrf_k: int = Field(
        default=60,
        gt=0,
    )

    # ==========================
    # Reranker配置
    # ==========================

    reranker_enabled: bool = False
    reranker_provider: str = "bailian"
    reranker_base_url: str | None = None
    reranker_model: str = "qwen3-rerank"
    reranker_api_key: str | None = None
    reranker_timeout: int = Field(default=30, gt=0)
    reranker_fail_open: bool = True
    reranker_instruct: str | None = None

    # ==========================
    # Authentication / JWT 配置
    # ==========================

    jwt_secret_key: str = Field(min_length=32)
    jwt_algorithm: Literal["HS256"] = "HS256"
    jwt_access_token_expire_minutes: int = Field(
        default=60,
        gt=0,
    )


    # ==========================
    # Storage 配置
    # ==========================

    storage_backend: Literal[
        "local",
        "minio",
    ] = "local"

    local_storage_dir: str = "uploads"

    minio_endpoint: str = "127.0.0.1:9000"
    minio_access_key: str | None = None
    minio_secret_key: str | None = None
    minio_bucket: str = "knowledge-assistant"
    minio_secure: bool = False

    # ==========================
    # Database配置
    # ==========================

    DATABASE_URL: str = (
        "sqlite:///./knowledge_assistant.db"
    )


    TEST_DATABASE_URL: str = (
        "sqlite:///./test.db"
    )

    database_pool_pre_ping: bool = True
    database_pool_recycle: int = Field(default=1800, gt=0)

    # ==========================
    # Redis / Celery 配置
    # ==========================

    redis_url: str = "redis://127.0.0.1:6379/0"
    celery_broker_url: str = "redis://127.0.0.1:6379/0"
    celery_result_backend: str = "redis://127.0.0.1:6379/1"
    celery_task_always_eager: bool = False
    celery_task_eager_propagates: bool = True
    celery_visibility_timeout: int = Field(default=3600, gt=0)

    processing_job_max_retries: int = Field(default=3, ge=0)
    processing_job_retry_base_delay: int = Field(default=2, gt=0)
    processing_job_retry_max_delay: int = Field(default=30, gt=0)
    processing_job_lease_seconds: int = Field(default=900, gt=0)


    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ==========================
    # Vector Store 配置
    # ==========================

    vector_store_backend: Literal[
        "database",
        "qdrant",
    ] = "database"

    qdrant_url: str = (
        "http://127.0.0.1:6333"
    )

    qdrant_api_key: str | None = None

    qdrant_collection_name: str = (
        "knowledge_assistant_chunks"
    )

    qdrant_timeout: int = Field(
        default=10,
        gt=0,
    )

    @model_validator(mode="after")
    def validate_runtime_security(self) -> "Settings":
        """校验跨域与生产环境敏感配置。"""
        if self.cors_allow_credentials and "*" in self.cors_allowed_origins:
            raise ValueError(
                "CORS wildcard cannot be used with credentials"
            )

        if self.app_environment != "production":
            return self

        if self.debug:
            raise ValueError("DEBUG must be disabled in production")

        if "*" in self.cors_allowed_origins:
            raise ValueError(
                "CORS wildcard is not allowed in production"
            )

        jwt_secret = self.jwt_secret_key.strip().lower()
        weak_jwt_markers = (
            "replace_with",
            "change_me",
            "changeme",
            "dev_secret",
        )
        if any(marker in jwt_secret for marker in weak_jwt_markers):
            raise ValueError(
                "JWT_SECRET_KEY must be replaced for production"
            )

        if self.storage_backend == "minio":
            if not self.minio_access_key or not self.minio_secret_key:
                raise ValueError(
                    "MinIO credentials are required in production"
                )

            minio_secret = self.minio_secret_key.strip().lower()
            if "dev_secret" in minio_secret or "change_me" in minio_secret:
                raise ValueError(
                    "MINIO_SECRET_KEY must be replaced for production"
                )

        return self

@lru_cache
def get_settings() -> Settings:
    """
    获取应用配置单例。
    """

    return Settings()