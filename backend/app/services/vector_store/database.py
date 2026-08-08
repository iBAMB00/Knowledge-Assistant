from collections.abc import Sequence
import math

from sqlalchemy.orm import Session

from app.repositories.chunk_embedding_repository import ChunkEmbeddingRepository
from app.schemas.vector_search_result import VectorSearchResult
from app.services.vector_store.base import ChunkRole, VectorStore


class DatabaseVectorStore(VectorStore):
    """
    基于关系数据库的向量检索实现。

    当前从chunk_embeddings表读取JSON向量，
    并在Python中计算余弦相似度。

    适用于MVP和小规模数据验证。

    不适用于：
    - 大规模向量检索
    - 高并发检索
    - 近似最近邻索引
    """

    def __init__(
        self,
        chunk_embedding_repository:
        ChunkEmbeddingRepository,
    ) -> None:
        """
        初始化数据库向量存储。
        """

        self.chunk_embedding_repository = (
            chunk_embedding_repository
        )

    def search(
        self,
        db: Session,
        query_vector: Sequence[float],
        embedding_model: str,
        top_k: int = 5,
        document_id: int | None = None,
        chunk_role: ChunkRole | None = None,
    ) -> list[VectorSearchResult]:
        """
        在数据库已有Chunk向量中执行Top-K检索。

        Args:
            db:
                数据库会话。
            query_vector:
                查询文本对应的向量。
            embedding_model:
                生成查询向量的模型名称。
            top_k:
                返回结果数量。
            document_id:
                可选文档过滤条件。

        Raises:
            ValueError:
                查询参数不合法。
            RuntimeError:
                数据库向量数据不完整或维度不一致。
        """

        normalized_query_vector = (
            self._normalize_query_vector(
                query_vector=query_vector,
            )
        )

        normalized_model = embedding_model.strip()

        if not normalized_model:
            raise ValueError(
                "embedding model cannot be empty"
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero"
            )

        if (
            document_id is not None
            and document_id <= 0
        ):
            raise ValueError(
                "document_id must be greater than zero"
            )

        query_norm = self._calculate_norm(
            vector=normalized_query_vector,
        )

        if query_norm == 0:
            raise ValueError(
                "query vector cannot be a zero vector"
            )

        candidates = (
            self.chunk_embedding_repository
            .find_search_candidates(
                db=db,
                embedding_model=normalized_model,
                document_id=document_id,
                chunk_role=chunk_role,
            )
        )

        results: list[VectorSearchResult] = []

        for embedding, chunk, document_content, document in candidates:
            candidate_vector = (
                self._normalize_candidate_vector(
                    vector=embedding.vector,
                    chunk_id=chunk.id,
                )
            )

            if (
                len(candidate_vector)
                != len(normalized_query_vector)
            ):
                raise RuntimeError(
                    "embedding vector dimension does not "
                    "match query vector: "
                    f"chunk_id={chunk.id}, "
                    f"query_dimension="
                    f"{len(normalized_query_vector)}, "
                    f"embedding_dimension="
                    f"{len(candidate_vector)}"
                )

            if (
                embedding.embedding_dimension
                != len(candidate_vector)
            ):
                raise RuntimeError(
                    "stored embedding dimension does not "
                    "match actual vector length: "
                    f"chunk_id={chunk.id}, "
                    f"stored_dimension="
                    f"{embedding.embedding_dimension}, "
                    f"actual_dimension="
                    f"{len(candidate_vector)}"
                )

            score = self._cosine_similarity(
                left=normalized_query_vector,
                right=candidate_vector,
                left_norm=query_norm,
            )

            results.append(
                VectorSearchResult(
                    document_id=document_content.document_id,
                    filename=document.filename,
                    chunk_id=chunk.id,
                    chunk_index=chunk.chunk_index,
                    content=chunk.content,
                    score=score,
                    parent_chunk_id=chunk.parent_chunk_id,
                )
            )

        results.sort(
            key=lambda result: (
                -result.score,
                result.chunk_id,
            )
        )

        return results[:top_k]

    @staticmethod
    def _normalize_query_vector(
        query_vector: Sequence[float],
    ) -> list[float]:
        """
        校验并转换查询向量。
        """

        if not query_vector:
            raise ValueError(
                "query vector cannot be empty"
            )

        normalized_vector: list[float] = []

        for value in query_vector:
            if (
                isinstance(value, bool)
                or not isinstance(
                    value,
                    (int, float),
                )
            ):
                raise ValueError(
                    "query vector must contain "
                    "only numeric values"
                )

            normalized_value = float(value)

            if not math.isfinite(
                normalized_value
            ):
                raise ValueError(
                    "query vector contains "
                    "non-finite value"
                )

            normalized_vector.append(
                normalized_value
            )

        return normalized_vector

    @staticmethod
    def _normalize_candidate_vector(
        vector: list[float] | None,
        chunk_id: int,
    ) -> list[float]:
        """
        校验数据库中保存的向量。
        """

        if not vector:
            raise RuntimeError(
                "stored embedding vector is empty: "
                f"chunk_id={chunk_id}"
            )

        normalized_vector: list[float] = []

        for value in vector:
            if (
                isinstance(value, bool)
                or not isinstance(
                    value,
                    (int, float),
                )
            ):
                raise RuntimeError(
                    "stored embedding vector contains "
                    "non-numeric value: "
                    f"chunk_id={chunk_id}"
                )

            normalized_value = float(value)

            if not math.isfinite(
                normalized_value
            ):
                raise RuntimeError(
                    "stored embedding vector contains "
                    "non-finite value: "
                    f"chunk_id={chunk_id}"
                )

            normalized_vector.append(
                normalized_value
            )

        return normalized_vector

    @staticmethod
    def _calculate_norm(
        vector: Sequence[float],
    ) -> float:
        """
        计算向量的欧几里得范数。
        """

        return math.sqrt(
            sum(
                value * value
                for value in vector
            )
        )

    @classmethod
    def _cosine_similarity(
        cls,
        left: Sequence[float],
        right: Sequence[float],
        left_norm: float,
    ) -> float:
        """
        计算两个向量的余弦相似度。
        """

        right_norm = cls._calculate_norm(
            vector=right,
        )

        if right_norm == 0:
            raise RuntimeError(
                "stored embedding vector "
                "cannot be a zero vector"
            )

        dot_product = sum(
            left_value * right_value
            for left_value, right_value
            in zip(
                left,
                right,
                strict=True,
            )
        )

        score = (
            dot_product
            / (left_norm * right_norm)
        )

        # 避免浮点误差导致结果略微超出[-1, 1]。
        return max(
            -1.0,
            min(
                1.0,
                score,
            ),
        )