from collections.abc import Sequence

from openai import OpenAI

from app.core.config import get_settings
from app.services.embedding.base import EmbeddingProvider


class VolcengineEmbeddingProvider(
    EmbeddingProvider,
):
    """
    火山方舟Embedding Provider。

    通过OpenAI兼容接口调用火山方舟文本向量化模型。

    负责：
    - 校验待向量化文本
    - 调用火山方舟Embedding API
    - 按输入顺序返回向量结果

    不负责：
    - 文档状态管理
    - Chunk状态管理
    - 向量持久化
    """

    def __init__(self) -> None:
        """
        初始化火山方舟Embedding客户端。
        """

        settings = get_settings()

        self.client = OpenAI(
            api_key=settings.embedding_api_key,
            base_url=settings.embedding_base_url,
            timeout=60.0,
        )

        self.model = settings.embedding_model

    @property
    def model_name(self) -> str:
        """
        返回当前Embedding模型或推理接入点名称。
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
                待向量化的文本列表。

        Returns:
            与输入顺序一致的向量列表。

        Raises:
            ValueError:
                输入中存在空文本。
            RuntimeError:
                API返回的向量数量与输入数量不一致。
        """

        if not texts:
            return []

        input_texts = list(texts)

        if any(
            not isinstance(text, str)
            or not text.strip()
            for text in input_texts
        ):
            raise ValueError(
                "embedding text cannot be empty"
            )

        response = self.client.embeddings.create(
            model=self.model,
            input=input_texts,
            encoding_format="float",
        )

        ordered_data = sorted(
            response.data,
            key=lambda item: item.index,
        )

        vectors = [
            item.embedding
            for item in ordered_data
        ]

        if len(vectors) != len(input_texts):
            raise RuntimeError(
                "embedding result count does not match input count"
            )

        return vectors

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
            查询文本对应的向量。
        """

        vectors = self.embed_documents([
            text,
        ])

        return vectors[0]