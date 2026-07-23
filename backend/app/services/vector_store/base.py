from abc import ABC, abstractmethod
from collections.abc import Sequence

from sqlalchemy.orm import Session

from app.schemas.vector_search_result import (
    VectorSearchResult,
)


class VectorStore(ABC):
    """
    向量存储抽象。

    定义统一的向量检索能力。

    不负责：
    - 查询文本向量化
    - Prompt组装
    - LLM调用
    """

    @abstractmethod
    def search(
        self,
        db: Session,
        query_vector: Sequence[float],
        embedding_model: str,
        top_k: int = 5,
        document_id: int | None = None,
    ) -> list[VectorSearchResult]:
        """
        根据查询向量返回相似度最高的文本切片。

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
        """