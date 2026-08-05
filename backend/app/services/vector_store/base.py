from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

from sqlalchemy.orm import Session

from app.schemas.vector_search_result import (
    VectorSearchResult,
)


@dataclass(frozen=True)
class VectorIndexRecord:
    """
    待写入向量索引的业务数据。

    不包含Qdrant等具体存储实现的专有类型。
    """

    chunk_id: int
    document_id: int
    chunk_index: int
    filename: str
    content: str
    embedding_model: str
    vector: Sequence[float]


class VectorStore(ABC):
    """
    向量检索抽象。

    负责：
    - 根据查询向量召回相关Chunk

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
        """



class VectorIndex(ABC):
    """
    外部向量索引写入抽象。

    SQL中的ChunkEmbedding仍然是向量事实数据，
    VectorIndex负责维护可重建的检索索引。
    """

    @abstractmethod
    def ensure_collection(self) -> None:
        """
        确保向量Collection存在且配置正确。
        """

    @abstractmethod
    def upsert(
        self,
        records: Sequence[VectorIndexRecord],
    ) -> None:
        """
        幂等写入或更新向量索引。
        """

    @abstractmethod
    def delete_by_document_id(
        self,
        document_id: int,
    ) -> None:
        """
        删除指定文档的全部向量索引。
        """