from functools import lru_cache
from typing import Literal

from pydantic import Field
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
        default=0.50,
        ge=-1.0,
        le=1.0,
    )

    retrieval_per_document_limit: int = Field(
        default=2,
        gt=0,
    )

    # ==========================
    # Database配置
    # ==========================

    DATABASE_URL: str = (
        "sqlite:///./knowledge_assistant.db"
    )


    TEST_DATABASE_URL: str = (
        "sqlite:///./test.db"
    )


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

@lru_cache
def get_settings() -> Settings:
    """
    获取应用配置单例。
    """

    return Settings()