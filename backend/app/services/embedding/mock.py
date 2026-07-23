from hashlib import sha256
from collections.abc import Sequence

from app.services.embedding.base import EmbeddingProvider


class MockEmbeddingProvider(EmbeddingProvider):
    """
    用于开发和测试的确定性Embedding实现。

    相同文本始终生成相同向量，
    不代表真实语义相似度。
    """

    def __init__(
        self,
        dimension: int = 8,
    ) -> None:
        """
        初始化Mock Provider。

        Args:
            dimension:
                生成向量的维度。
        """

        if dimension <= 0:
            raise ValueError(
                "dimension must be greater than 0"
            )

        self.dimension = dimension

    @property
    def model_name(self) -> str:
        """
        返回Mock模型名称。
        """

        return "mock-sha256"

    def embed_documents(
        self,
        texts: Sequence[str],
    ) -> list[list[float]]:
        """
        批量生成确定性向量。
        """

        return [
            self._embed(text)
            for text in texts
        ]

    def embed_query(
        self,
        text: str,
    ) -> list[float]:
        """
        生成查询文本向量。
        """

        return self._embed(text)

    def _embed(
        self,
        text: str,
    ) -> list[float]:
        """
        使用SHA-256稳定生成指定维度向量。
        """

        if not text.strip():
            raise ValueError(
                "embedding text cannot be empty"
            )

        values: list[float] = []
        counter = 0

        while len(values) < self.dimension:
            digest = sha256(
                f"{counter}:{text}".encode("utf-8")
            ).digest()

            values.extend(
                round(
                    byte / 127.5 - 1.0,
                    8,
                )
                for byte in digest
            )

            counter += 1

        return values[:self.dimension]