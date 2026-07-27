from sqlalchemy.orm import Session

from app.schemas.vector_search_result import VectorSearchResult
from app.services.embedding.base import EmbeddingProvider
from app.services.vector_store.base import VectorStore


class RetrievalService:
    """
    知识库检索编排服务。

    负责：
    - 校验查询文本和检索参数
    - 生成查询文本向量
    - 调用VectorStore执行相似度检索
    - 根据相似度阈值过滤结果

    不负责：
    - 文档向量化
    - Prompt组装
    - LLM调用
    - HTTP响应处理
    """

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
        default_top_k: int = 5,
        default_score_threshold: float = -1.0,
    ) -> None:
        """
        初始化检索服务。

        Args:
            embedding_provider:
                查询文本使用的Embedding Provider。
            vector_store:
                向量存储实现。
            default_top_k:
                默认返回结果数量。
            default_score_threshold:
                默认余弦相似度阈值，范围为[-1, 1]。
        """

        self._validate_top_k(default_top_k)
        self._validate_score_threshold(
            default_score_threshold
        )

        self.embedding_provider = embedding_provider
        self.vector_store = vector_store
        self.default_top_k = default_top_k
        self.default_score_threshold = (
            default_score_threshold
        )

    def retrieve(
        self,
        db: Session,
        query: str,
        top_k: int | None = None,
        score_threshold: float | None = None,
        document_id: int | None = None,
    ) -> list[VectorSearchResult]:
        """
        根据用户问题检索相关文本切片。

        Args:
            db:
                数据库会话。
            query:
                用户查询文本。
            top_k:
                可选返回结果数量。
            score_threshold:
                可选余弦相似度阈值。
            document_id:
                可选文档过滤条件。

        Returns:
            按相似度从高到低排列的检索结果。
        """

        normalized_query = query.strip()

        if not normalized_query:
            raise ValueError(
                "query cannot be empty"
            )

        resolved_top_k = (
            self.default_top_k
            if top_k is None
            else top_k
        )

        resolved_score_threshold = (
            self.default_score_threshold
            if score_threshold is None
            else score_threshold
        )

        self._validate_top_k(resolved_top_k)
        self._validate_score_threshold(
            resolved_score_threshold
        )

        query_vector = (
            self.embedding_provider.embed_query(
                normalized_query
            )
        )

        results = self.vector_store.search(
            db=db,
            query_vector=query_vector,
            embedding_model=(
                self.embedding_provider.model_name
            ),
            top_k=resolved_top_k,
            document_id=document_id,
        )

        return [
            result
            for result in results
            if result.score
            >= resolved_score_threshold
        ]

    @staticmethod
    def _validate_top_k(top_k: int) -> None:
        """
        校验Top-K参数。
        """

        if (
            isinstance(top_k, bool)
            or not isinstance(top_k, int)
            or top_k <= 0
        ):
            raise ValueError(
                "top_k must be a positive integer"
            )

    @staticmethod
    def _validate_score_threshold(
        score_threshold: float,
    ) -> None:
        """
        校验余弦相似度阈值。
        """

        if (
            isinstance(score_threshold, bool)
            or not isinstance(
                score_threshold,
                (int, float),
            )
        ):
            raise ValueError(
                "score_threshold must be numeric"
            )

        if not -1.0 <= float(score_threshold) <= 1.0:
            raise ValueError(
                "score_threshold must be between "
                "-1 and 1"
            )