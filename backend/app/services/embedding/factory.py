from app.core.config import get_settings
from app.services.embedding.bailian_embedding import (
    BailianEmbeddingProvider,
)
from app.services.embedding.base import EmbeddingProvider
from app.services.embedding.mock import MockEmbeddingProvider
from app.services.embedding.volcengine_embedding import (
    VolcengineEmbeddingProvider,
)


class EmbeddingFactory:
    """
    Embedding Provider工厂。

    根据应用配置创建具体Provider实现。
    """

    @staticmethod
    def create() -> EmbeddingProvider:
        """
        根据embedding_provider创建Provider。
        """

        settings = get_settings()

        provider = (
            settings.embedding_provider
            .strip()
            .lower()
        )

        if provider == "mock":
            return MockEmbeddingProvider(
                dimension=(
                    settings.embedding_dimension
                    or 8
                ),
            )

        if provider == "volcengine":
            return VolcengineEmbeddingProvider()

        if provider == "bailian":
            return BailianEmbeddingProvider(
                api_key=EmbeddingFactory._require_value(
                    value=settings.embedding_api_key,
                    field_name="embedding_api_key",
                ),
                base_url=EmbeddingFactory._require_value(
                    value=settings.embedding_base_url,
                    field_name="embedding_base_url",
                ),
                model=EmbeddingFactory._require_value(
                    value=settings.embedding_model,
                    field_name="embedding_model",
                ),
                dimension=settings.embedding_dimension,
            )

        raise ValueError(
            "unsupported embedding provider: "
            f"{provider}"
        )

    @staticmethod
    def _require_value(
        value: str | None,
        field_name: str,
    ) -> str:
        """
        校验Provider必需配置。
        """

        if value is None or not value.strip():
            raise ValueError(
                f"{field_name} is required"
            )

        return value.strip()