from app.core.config import get_settings
from app.services.embedding.base import (
    EmbeddingProvider,
)
from app.services.embedding.mock import (
    MockEmbeddingProvider,
)
from app.services.embedding.volcengine_embedding import (
    VolcengineEmbeddingProvider,
)


class EmbeddingFactory:
    """
    Embedding Provider 工厂。

    根据配置创建对应的Embedding实现。

    不负责：
    - Embedding业务流程
    - 向量存储
    - 状态管理
    """

    @staticmethod
    def create() -> EmbeddingProvider:
        """
        根据配置创建Embedding Provider。

        Returns:
            EmbeddingProvider实例。

        Raises:
            ValueError:
                不支持的Embedding Provider。
        """

        settings = get_settings()

        provider = (
            settings.embedding_provider
            .lower()
        )

        if provider == "mock":
            return MockEmbeddingProvider(
                dimension=(
                    settings.embedding_dimension
                ),
            )

        if provider == "volcengine":
            return VolcengineEmbeddingProvider()

        raise ValueError(
            f"unsupported embedding provider: {provider}"
        )