from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):

    app_name: str = "Secure Assistant AI Agent"

    debug: bool = False

    model_provider: str
    model_base_url: str
    model_name: str
    model_api_key: str


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
    return Settings()