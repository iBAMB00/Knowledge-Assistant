from functools import lru_cache

from pydantic_settings import (
    BaseSettings,
    SettingsConfigDict,
)


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

    embedding_provider: str = (
        "volcengine"
    )

    embedding_base_url: str

    embedding_model: str

    embedding_api_key: str

    embedding_dimension: int = 1024


    # ==========================
    # Database配置
    # ==========================

    DATABASE_URL: str = (
        "sqlite:///./secure_assistant.db"
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


@lru_cache
def get_settings() -> Settings:
    """
    获取应用配置单例。
    """

    return Settings()