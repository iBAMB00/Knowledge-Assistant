from collections.abc import Sequence
import logging
from math import isfinite
from typing import Literal

from sqlalchemy.orm import Session

from app.repositories.document_chunk_repository import DocumentChunkRepository
from app.schemas.vector_search_result import VectorSearchResult
from app.services.bm25_retrieval_service import BM25RetrievalService
from app.services.embedding.base import EmbeddingProvider
from app.services.rrf_fusion_service import RRFFusionService
from app.services.reranker.base import RerankerProvider
from app.services.vector_store.base import ChunkRole, VectorStore


logger = logging.getLogger(__name__)


RetrievalMode = Literal[
    "baseline",
    "optimized",
]


class RetrievalService:
    """
    知识库检索编排服务。

    负责：
    - 校验查询文本和检索参数
    - 生成查询文本向量
    - 执行Baseline或Optimized检索
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

    BASELINE_MODE = "baseline"
    OPTIMIZED_MODE = "optimized"

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
        default_top_k: int = 5,
        default_candidate_k: int = 20,
        default_score_threshold: float = -1.0,
        default_per_document_limit: int = 2,
        document_chunk_repository: DocumentChunkRepository | None = None,
        parent_child_enabled: bool = False,
        bm25_retriever: BM25RetrievalService | None = None,
        rrf_fusion_service: RRFFusionService | None = None,
        hybrid_enabled: bool = False,
        reranker: RerankerProvider | None = None,
        reranker_enabled: bool = False,
        reranker_fail_open: bool = True,
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
        self.document_chunk_repository = document_chunk_repository
        self.parent_child_enabled = parent_child_enabled
        self.bm25_retriever = bm25_retriever
        self.rrf_fusion_service = rrf_fusion_service
        self.hybrid_enabled = hybrid_enabled
        self.reranker = reranker
        self.reranker_enabled = reranker_enabled
        self.reranker_fail_open = reranker_fail_open

        if self.hybrid_enabled and (
            self.bm25_retriever is None
            or self.rrf_fusion_service is None
        ):
            raise ValueError(
                "bm25_retriever and rrf_fusion_service are required "
                "when hybrid_enabled is true"
            )

        if self.reranker_enabled and self.reranker is None:
            raise ValueError(
                "reranker is required when reranker_enabled is true"
            )

        if (
            self.parent_child_enabled
            and self.document_chunk_repository is None
        ):
            raise ValueError(
                "document_chunk_repository is required "
                "when parent_child_enabled is true"
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
        retrieval_mode: RetrievalMode = "optimized",
    ) -> list[VectorSearchResult]:
        """
        根据用户问题检索相关文本切片。

        该公开方法负责生成查询向量，随后复用
        retrieve_by_vector执行具体检索。
        """

        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError(
                "query cannot be empty"
            )

        query_vector = self.embed_query(normalized_query)

        return self.retrieve_by_vector(
            db=db,
            query_vector=query_vector,
            top_k=top_k,
            candidate_k=candidate_k,
            score_threshold=score_threshold,
            per_document_limit=per_document_limit,
            document_id=document_id,
            retrieval_mode=retrieval_mode,
            query_text=normalized_query,
        )

    def embed_query(
        self,
        query: str,
    ) -> list[float]:
        """校验查询文本并生成查询向量。"""

        normalized_query = query.strip()

        if not normalized_query:
            raise ValueError(
                "query cannot be empty"
            )

        query_vector = (
            self.embedding_provider.embed_query(
                normalized_query
            )
        )

        return self._normalize_query_vector(
            query_vector
        )

    def retrieve_by_vector(
        self,
        db: Session,
        query_vector: Sequence[float],
        top_k: int | None = None,
        candidate_k: int | None = None,
        score_threshold: float | None = None,
        per_document_limit: int | None = None,
        document_id: int | None = None,
        retrieval_mode: RetrievalMode = "optimized",
        query_text: str | None = None,
    ) -> list[VectorSearchResult]:
        """
        使用已经生成的查询向量执行检索。

        评估器可让Baseline和Optimized共用同一查询向量，
        避免重复调用Embedding并减少网络波动对对比结果的干扰。
        """

        normalized_query_vector = (
            self._normalize_query_vector(
                query_vector
            )
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
        self._validate_score_threshold(
            resolved_score_threshold
        )
        self._validate_retrieval_mode(
            retrieval_mode
        )

        if retrieval_mode == self.OPTIMIZED_MODE:
            self._validate_positive_integer(
                value=resolved_candidate_k,
                field_name="candidate_k",
            )
            self._validate_positive_integer(
                value=resolved_per_document_limit,
                field_name="per_document_limit",
            )
            self._validate_candidate_k(
                candidate_k=resolved_candidate_k,
                top_k=resolved_top_k,
            )

        if retrieval_mode == self.BASELINE_MODE:
            return self._retrieve_baseline(
                db=db,
                query_vector=normalized_query_vector,
                top_k=resolved_top_k,
                score_threshold=(
                    resolved_score_threshold
                ),
                document_id=document_id,
            )

        return self._retrieve_optimized(
            db=db,
            query_vector=normalized_query_vector,
            top_k=resolved_top_k,
            candidate_k=resolved_candidate_k,
            score_threshold=(
                resolved_score_threshold
            ),
            per_document_limit=(
                resolved_per_document_limit
            ),
            document_id=document_id,
            query_text=query_text,
        )

    def _retrieve_baseline(
        self,
        db: Session,
        query_vector: Sequence[float],
        top_k: int,
        score_threshold: float,
        document_id: int | None,
    ) -> list[VectorSearchResult]:
        """
        执行原始Top-K检索。

        Baseline严格保留原始行为：
        只进行Top-K向量召回和分数过滤。
        """

        results = self._search_vector_store(
            db=db,
            query_vector=query_vector,
            top_k=top_k,
            document_id=document_id,
            chunk_role=(
                "parent"
                if self.parent_child_enabled
                else None
            ),
        )

        return self._filter_by_score(
            results=results,
            score_threshold=score_threshold,
        )

    def _retrieve_optimized(
        self,
        db: Session,
        query_vector: Sequence[float],
        top_k: int,
        candidate_k: int,
        score_threshold: float,
        per_document_limit: int,
        document_id: int | None,
        query_text: str | None,
    ) -> list[VectorSearchResult]:
        """
        执行候选扩召回和多文档优化检索。
        """

        chunk_role: ChunkRole | None = (
            "child"
            if self.parent_child_enabled
            else None
        )

        dense_candidates = self._search_vector_store(
            db=db,
            query_vector=query_vector,
            top_k=candidate_k,
            document_id=document_id,
            chunk_role=chunk_role,
        )

        filtered_dense_results = (
            self._filter_and_deduplicate(
                results=dense_candidates,
                score_threshold=score_threshold,
            )
        )

        filtered_results = filtered_dense_results

        if self.hybrid_enabled and query_text is not None:
            bm25_retriever = self.bm25_retriever
            rrf_fusion_service = self.rrf_fusion_service

            if bm25_retriever is None or rrf_fusion_service is None:
                raise RuntimeError(
                    "hybrid retrieval dependencies are not configured"
                )

            lexical_results = bm25_retriever.search(
                db=db,
                query=query_text,
                top_k=candidate_k,
                document_id=document_id,
                chunk_role=chunk_role,
            )

            filtered_results = rrf_fusion_service.fuse(
                rankings=[
                    filtered_dense_results,
                    lexical_results,
                ],
                top_k=candidate_k,
            )

        if (
            self.reranker_enabled
            and query_text is not None
            and filtered_results
        ):
            filtered_results = self._rerank_candidates(
                query=query_text,
                results=filtered_results,
            )

        if self.parent_child_enabled:
            filtered_results = self._expand_parent_contexts(
                db=db,
                results=filtered_results,
            )
            filtered_results = (
                self._deduplicate_parent_contexts(
                    filtered_results
                )
            )

        # 指定单文档时，不执行多文档平衡。
        if document_id is not None:
            return filtered_results[:top_k]

        return self._balance_documents(
            results=filtered_results,
            top_k=top_k,
            per_document_limit=(
                per_document_limit
            ),
        )

    def _search_vector_store(
        self,
        db: Session,
        query_vector: Sequence[float],
        top_k: int,
        document_id: int | None,
        chunk_role: ChunkRole | None,
    ) -> list[VectorSearchResult]:
        """执行向量检索，并在Parent-Child模式下限定Chunk角色。"""

        if chunk_role is None:
            return self.vector_store.search(
                db=db,
                query_vector=query_vector,
                embedding_model=self.embedding_provider.model_name,
                top_k=top_k,
                document_id=document_id,
            )

        return self.vector_store.search(
            db=db,
            query_vector=query_vector,
            embedding_model=self.embedding_provider.model_name,
            top_k=top_k,
            document_id=document_id,
            chunk_role=chunk_role,
        )

    def _rerank_candidates(
        self,
        query: str,
        results: list[VectorSearchResult],
    ) -> list[VectorSearchResult]:
        """使用重排序模型重新评估候选 Child 的相关性。"""

        reranker = self.reranker
        if reranker is None:
            return results

        try:
            reranked_items = reranker.rerank(
                query=query,
                documents=[result.content for result in results],
                top_n=len(results),
            )
        except Exception as exc:
            if not self.reranker_fail_open:
                raise

            logger.warning(
                "reranker failed, fallback to pre-rerank ranking: "
                "model=%s, error_type=%s",
                reranker.model_name,
                type(exc).__name__,
            )
            return results

        logger.info(
            "reranker completed: model=%s, candidates=%d, returned=%d",
            reranker.model_name,
            len(results),
            len(reranked_items),
        )

        reranked_results: list[VectorSearchResult] = []
        for item in reranked_items:
            candidate = results[item.index]
            reranked_results.append(
                candidate.model_copy(
                    update={"score": item.score}
                )
            )

        return reranked_results

    def _expand_parent_contexts(
        self,
        db: Session,
        results: list[VectorSearchResult],
    ) -> list[VectorSearchResult]:
        """保留Child命中信息，同时把返回正文扩展为Parent内容。"""

        repository = self.document_chunk_repository
        if repository is None:
            return results

        parent_ids = sorted({
            result.parent_chunk_id
            for result in results
            if result.parent_chunk_id is not None
        })

        if not parent_ids:
            return []

        parents = repository.find_by_ids(
            db=db,
            chunk_ids=parent_ids,
        )
        parents_by_id = {parent.id: parent for parent in parents}

        expanded: list[VectorSearchResult] = []

        for result in results:
            parent_chunk_id = result.parent_chunk_id
            if parent_chunk_id is None:
                continue

            parent = parents_by_id.get(parent_chunk_id)
            if parent is None:
                logger.warning(
                    "parent chunk missing during retrieval: "
                    "child_chunk_id=%s, parent_chunk_id=%s",
                    result.chunk_id,
                    parent_chunk_id,
                )
                continue

            expanded.append(
                result.model_copy(
                    update={"content": parent.content}
                )
            )

        return expanded

    @staticmethod
    def _deduplicate_parent_contexts(
        results: list[VectorSearchResult],
    ) -> list[VectorSearchResult]:
        """同一Parent被多个Child命中时只保留最高分结果。"""

        deduplicated: list[VectorSearchResult] = []
        seen_parent_ids: set[int] = set()

        for result in results:
            parent_chunk_id = result.parent_chunk_id
            if parent_chunk_id is None:
                continue
            if parent_chunk_id in seen_parent_ids:
                continue

            seen_parent_ids.add(parent_chunk_id)
            deduplicated.append(result)

        return deduplicated

    @staticmethod
    def _filter_by_score(
        results: list[VectorSearchResult],
        score_threshold: float,
    ) -> list[VectorSearchResult]:
        """
        只根据相似度阈值过滤结果。

        用于复现原始Baseline行为。
        """

        return [
            result
            for result in results
            if result.score >= score_threshold
        ]

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
    def _normalize_query_vector(
        query_vector: Sequence[float],
    ) -> list[float]:
        """规范化并校验查询向量。"""

        normalized_vector = [
            float(value)
            for value in query_vector
        ]

        if not normalized_vector:
            raise ValueError(
                "query_vector cannot be empty"
            )

        if any(
            not isfinite(value)
            for value in normalized_vector
        ):
            raise ValueError(
                "query_vector must contain finite values"
            )

        return normalized_vector

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

    @classmethod
    def _validate_retrieval_mode(
        cls,
        retrieval_mode: str,
    ) -> None:
        """
        校验检索模式。
        """

        if retrieval_mode not in {
            cls.BASELINE_MODE,
            cls.OPTIMIZED_MODE,
        }:
            raise ValueError(
                "retrieval_mode must be either "
                "'baseline' or 'optimized'"
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