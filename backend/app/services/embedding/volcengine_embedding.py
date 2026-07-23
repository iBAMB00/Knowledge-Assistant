from collections.abc import Sequence

from openai import OpenAI

from app.core.config import get_settings
from app.services.embedding.base import (
    EmbeddingProvider,
)


class VolcengineEmbeddingProvider(
    EmbeddingProvider,
):
    """
    火山方舟Embedding实现。

    基于OpenAI兼容接口调用Doubao Embedding模型。

    负责：
    - 调用火山方舟Embedding API
    - 转换响应结果
    - 提供模型信息

    不负责：
    - 文档处理流程
    - Chunk状态管理
    - 向量持久化
    """

    def __init__(
        self,
    ) -> None:
        """
        初始化Embedding客户端。
        """

        settings = get_settings()

        self.client = OpenAI(
            api_key=settings.embedding_api_key,
            base_url=settings.embedding_base_url,
        )

        self.model = settings.embedding_model
        self.dimension = (
            settings.embedding_dimension
        )


    @property
    def model_name(
        self,
    ) -> str:
        """
        返回当前Embedding模型名称。
        """

        return self.model


    def embed_documents(
        self,
        texts: Sequence[str],
    ) -> list[list[float]]:
        """
        批量生成文档向量。

        Args:
            texts:
                文档切片文本。

        Returns:
            与输入顺序一致的向量列表。
        """

        if not texts:
            return []

        response = (
            self.client.embeddings.create(
                model=self.model,
                input=list(texts),
                dimensions=self.dimension,
            )
        )

        return [
            item.embedding
            for item in response.data
        ]


    def embed_query(
        self,
        text: str,
    ) -> list[float]:
        """
        生成查询文本向量。

        Args:
            text:
                用户查询文本。

        Returns:
            查询向量。
        """

        if not text.strip():
            raise ValueError(
                "query text cannot be empty"
            )

        response = (
            self.client.embeddings.create(
                model=self.model,
                input=text,
                dimensions=self.dimension,
            )
        )

        return response.data[0].embedding