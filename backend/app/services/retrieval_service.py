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
    - 召回候选向量结果
    - 根据相似度阈值过滤结果
    - 过滤空内容和重复内容
    - 平衡不同文档的召回数量
    - 返回最终Top-K结果

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
        default_candidate_k: int = 20,
        default_score_threshold: float = -1.0,
        default_per_document_limit: int = 2,
    ) -> None:
        """
        初始化检索服务。
        """

        self._validate_positive_integer(
            value=default_top_k,
            field_name="top_k",
        )
        self._validate_positive_integer(
            value=default_candidate_k,
            field_name="candidate_k",
        )
        self._validate_positive_integer(
            value=default_per_document_limit,
            field_name="per_document_limit",
        )
        self._validate_score_threshold(
            default_score_threshold
        )
        self._validate_candidate_k(
            candidate_k=default_candidate_k,
            top_k=default_top_k,
        )

        self.embedding_provider = embedding_provider
        self.vector_store = vector_store

        self.default_top_k = default_top_k
        self.default_candidate_k = default_candidate_k
        self.default_score_threshold = (
            default_score_threshold
        )
        self.default_per_document_limit = (
            default_per_document_limit
        )

    def retrieve(
        self,
        db: Session,
        query: str,
        top_k: int | None = None,
        candidate_k: int | None = None,
        score_threshold: float | None = None,
        per_document_limit: int | None = None,
        document_id: int | None = None,
    ) -> list[VectorSearchResult]:
        """
        根据用户问题检索相关文本切片。
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

        resolved_candidate_k = (
            self.default_candidate_k
            if candidate_k is None
            else candidate_k
        )

        resolved_score_threshold = (
            self.default_score_threshold
            if score_threshold is None
            else score_threshold
        )

        resolved_per_document_limit = (
            self.default_per_document_limit
            if per_document_limit is None
            else per_document_limit
        )

        self._validate_positive_integer(
            value=resolved_top_k,
            field_name="top_k",
        )
        self._validate_positive_integer(
            value=resolved_candidate_k,
            field_name="candidate_k",
        )
        self._validate_positive_integer(
            value=resolved_per_document_limit,
            field_name="per_document_limit",
        )
        self._validate_score_threshold(
            resolved_score_threshold
        )
        self._validate_candidate_k(
            candidate_k=resolved_candidate_k,
            top_k=resolved_top_k,
        )

        query_vector = (
            self.embedding_provider.embed_query(
                normalized_query
            )
        )

        candidates = self.vector_store.search(
            db=db,
            query_vector=query_vector,
            embedding_model=(
                self.embedding_provider.model_name
            ),
            top_k=resolved_candidate_k,
            document_id=document_id,
        )

        filtered_results = (
            self._filter_and_deduplicate(
                results=candidates,
                score_threshold=(
                    resolved_score_threshold
                ),
            )
        )

        # 指定单文档检索时，不需要执行多文档平衡。
        if document_id is not None:
            return filtered_results[
                :resolved_top_k
            ]

        return self._balance_documents(
            results=filtered_results,
            top_k=resolved_top_k,
            per_document_limit=(
                resolved_per_document_limit
            ),
        )

    @staticmethod
    def _filter_and_deduplicate(
        results: list[VectorSearchResult],
        score_threshold: float,
    ) -> list[VectorSearchResult]:
        """
        过滤低分、空内容和重复检索结果。

        相同文档内标准化内容完全一致的Chunk，
        只保留排序更靠前的一条。
        """

        filtered_results: list[
            VectorSearchResult
        ] = []

        seen_chunk_ids: set[int] = set()

        seen_content_keys: set[
            tuple[int, str]
        ] = set()

        for result in results:
            if result.score < score_threshold:
                continue

            if result.chunk_id in seen_chunk_ids:
                continue

            normalized_content = " ".join(
                result.content.split()
            )

            if not normalized_content:
                continue

            content_key = (
                result.document_id,
                normalized_content,
            )

            if content_key in seen_content_keys:
                continue

            seen_chunk_ids.add(result.chunk_id)
            seen_content_keys.add(content_key)
            filtered_results.append(result)

        return filtered_results

    @staticmethod
    def _balance_documents(
        results: list[VectorSearchResult],
        top_k: int,
        per_document_limit: int,
    ) -> list[VectorSearchResult]:
        """
        平衡不同文档的检索结果。

        第一轮优先限制每个文档的结果数量。
        如果结果不足Top-K，再使用被限制的高分结果补足。
        """

        selected: list[
            tuple[int, VectorSearchResult]
        ] = []

        overflow: list[
            tuple[int, VectorSearchResult]
        ] = []

        document_counts: dict[int, int] = {}

        for index, result in enumerate(results):
            current_count = document_counts.get(
                result.document_id,
                0,
            )

            if current_count < per_document_limit:
                selected.append(
                    (index, result)
                )

                document_counts[result.document_id] = (
                    current_count + 1
                )

                if len(selected) >= top_k:
                    break

            else:
                overflow.append(
                    (index, result)
                )

        if len(selected) < top_k:
            remaining_count = (
                top_k - len(selected)
            )

            selected.extend(
                overflow[:remaining_count]
            )

        # 回填后恢复原始相关度顺序。
        selected.sort(
            key=lambda item: item[0]
        )

        return [
            result
            for _, result in selected[:top_k]
        ]

    @staticmethod
    def _validate_positive_integer(
        value: int,
        field_name: str,
    ) -> None:
        """
        校验正整数参数。
        """

        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
        ):
            raise ValueError(
                f"{field_name} must be "
                "a positive integer"
            )

    @staticmethod
    def _validate_candidate_k(
        candidate_k: int,
        top_k: int,
    ) -> None:
        """
        校验候选数量不能小于最终结果数量。
        """

        if candidate_k < top_k:
            raise ValueError(
                "candidate_k must be greater "
                "than or equal to top_k"
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