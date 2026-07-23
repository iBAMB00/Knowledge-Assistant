import math
from collections.abc import Sequence

from openai import OpenAI

from app.services.embedding.base import EmbeddingProvider


class BailianEmbeddingProvider(EmbeddingProvider):
    """
    阿里云百炼Embedding Provider。

    通过百炼OpenAI兼容接口调用文本向量模型。

    负责：
    - 校验文本输入
    - 按接口限制进行批量拆分
    - 调用百炼Embedding API
    - 校验并返回向量结果

    不负责：
    - 文档状态管理
    - Chunk状态管理
    - 数据库事务
    - 向量持久化
    """

    CUSTOM_DIMENSION_MODELS = frozenset({
        "text-embedding-v3",
        "text-embedding-v4",
    })

    MAX_REQUEST_BATCH_SIZE = 10

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        dimension: int | None = None,
        client: OpenAI | None = None,
    ) -> None:
        """
        初始化百炼Embedding Provider。

        Args:
            api_key:
                百炼API Key。
            base_url:
                百炼OpenAI兼容接口基础地址。
            model:
                Embedding模型名称。
            dimension:
                期望的向量维度。
            client:
                可选OpenAI客户端，主要用于测试注入。
        """

        normalized_api_key = api_key.strip()
        normalized_base_url = base_url.strip()
        normalized_model = model.strip()

        if not normalized_api_key:
            raise ValueError(
                "embedding api key cannot be empty"
            )

        if not normalized_base_url:
            raise ValueError(
                "embedding base url cannot be empty"
            )

        if not normalized_model:
            raise ValueError(
                "embedding model cannot be empty"
            )

        if dimension is not None and dimension <= 0:
            raise ValueError(
                "embedding dimension must be greater than zero"
            )

        self._model = normalized_model
        self._dimension = dimension

        self._client = client or OpenAI(
            api_key=normalized_api_key,
            base_url=normalized_base_url,
            timeout=60.0,
            max_retries=2,
        )

    @property
    def model_name(self) -> str:
        """
        返回当前Embedding模型名称。
        """

        return self._model

    def embed_documents(
        self,
        texts: Sequence[str],
    ) -> list[list[float]]:
        """
        批量生成文档向量。

        输入数量超过百炼单次接口限制时，
        Provider会自动拆分成多个请求。

        Args:
            texts:
                待向量化文本。

        Returns:
            与输入顺序一致的向量列表。

        Raises:
            ValueError:
                输入包含空文本。
            RuntimeError:
                API响应数量或向量维度不符合预期。
        """

        normalized_texts = self._normalize_texts(
            texts=texts,
        )

        if not normalized_texts:
            return []

        vectors: list[list[float]] = []

        for start in range(
            0,
            len(normalized_texts),
            self.MAX_REQUEST_BATCH_SIZE,
        ):
            batch = normalized_texts[
                start:
                start + self.MAX_REQUEST_BATCH_SIZE
            ]

            batch_vectors = self._embed_batch(
                texts=batch,
            )

            vectors.extend(batch_vectors)

        self._validate_vectors(
            vectors=vectors,
            expected_count=len(normalized_texts),
        )

        return vectors

    def embed_query(
        self,
        text: str,
    ) -> list[float]:
        """
        生成查询文本向量。

        查询和文档必须使用相同模型及相同维度，
        才能进行向量相似度计算。
        """

        vectors = self.embed_documents([
            text,
        ])

        return vectors[0]

    def _embed_batch(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        """
        调用一次百炼Embedding接口。
        """

        if (
            self._dimension is not None
            and self._supports_custom_dimension()
        ):
            response = self._client.embeddings.create(
                model=self._model,
                input=texts,
                dimensions=self._dimension,
                encoding_format="float",
            )
        else:
            response = self._client.embeddings.create(
                model=self._model,
                input=texts,
                encoding_format="float",
            )

        ordered_data = sorted(
            response.data,
            key=lambda item: item.index,
        )

        vectors = [
            [
                float(value)
                for value in item.embedding
            ]
            for item in ordered_data
        ]

        if len(vectors) != len(texts):
            raise RuntimeError(
                "embedding result count does not match batch input count"
            )

        return vectors

    def _supports_custom_dimension(self) -> bool:
        """
        判断当前模型是否支持dimensions请求参数。
        """

        return (
            self._model
            in self.CUSTOM_DIMENSION_MODELS
        )

    @staticmethod
    def _normalize_texts(
        texts: Sequence[str],
    ) -> list[str]:
        """
        校验并规范化文本。
        """

        normalized_texts: list[str] = []

        for text in texts:
            if not isinstance(text, str):
                raise ValueError(
                    "embedding text must be a string"
                )

            normalized_text = text.strip()

            if not normalized_text:
                raise ValueError(
                    "embedding text cannot be empty"
                )

            normalized_texts.append(
                normalized_text
            )

        return normalized_texts

    def _validate_vectors(
        self,
        vectors: list[list[float]],
        expected_count: int,
    ) -> None:
        """
        校验完整向量结果。
        """

        if len(vectors) != expected_count:
            raise RuntimeError(
                "embedding result count does not match input count"
            )

        if not vectors:
            return

        actual_dimensions = {
            len(vector)
            for vector in vectors
        }

        if 0 in actual_dimensions:
            raise RuntimeError(
                "embedding vector cannot be empty"
            )

        if len(actual_dimensions) != 1:
            raise RuntimeError(
                "embedding vector dimensions are inconsistent"
            )

        actual_dimension = next(
            iter(actual_dimensions)
        )

        if (
            self._dimension is not None
            and actual_dimension != self._dimension
        ):
            raise RuntimeError(
                "embedding vector dimension does not match configuration: "
                f"expected {self._dimension}, "
                f"actual {actual_dimension}"
            )

        if any(
            not math.isfinite(value)
            for vector in vectors
            for value in vector
        ):
            raise RuntimeError(
                "embedding vector contains non-finite value"
            )