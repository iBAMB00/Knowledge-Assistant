from collections.abc import Sequence
import math

from qdrant_client import QdrantClient, models
from sqlalchemy.orm import Session

from app.schemas.vector_search_result import VectorSearchResult
from app.services.vector_store.base import VectorIndex, VectorIndexRecord, VectorStore


class QdrantVectorStore(
    VectorStore,
    VectorIndex,
):
    """
    基于Qdrant的Dense向量存储。

    同时负责：
    - Qdrant Collection初始化
    - 向量Point写入
    - Dense向量检索
    - 文档级索引删除

    SQL中的ChunkEmbedding仍然是可恢复的数据来源。
    """

    DOCUMENT_ID_FIELD = "document_id"
    EMBEDDING_MODEL_FIELD = "embedding_model"

    def __init__(
        self,
        client: QdrantClient,
        collection_name: str,
        vector_size: int,
    ) -> None:
        """初始化Qdrant向量存储。"""

        normalized_collection_name = (
            collection_name.strip()
        )

        if not normalized_collection_name:
            raise ValueError(
                "collection_name cannot be empty"
            )

        if vector_size <= 0:
            raise ValueError(
                "vector_size must be greater than zero"
            )

        self.client = client
        self.collection_name = (
            normalized_collection_name
        )
        self.vector_size = vector_size
        self._collection_ready = False

    def ensure_collection(self) -> None:
        """
        确保Collection存在且向量配置正确。

        不自动删除或重建配置不匹配的Collection。
        """

        if self._collection_ready:
            return

        collection_exists = (
            self.client.collection_exists(
                collection_name=(
                    self.collection_name
                ),
            )
        )

        if not collection_exists:
            self.client.create_collection(
                collection_name=(
                    self.collection_name
                ),
                vectors_config=models.VectorParams(
                    size=self.vector_size,
                    distance=models.Distance.COSINE,
                ),
            )
        else:
            self._validate_collection()

        self._ensure_payload_indexes()
        self._collection_ready = True

    def upsert(
        self,
        records: Sequence[VectorIndexRecord],
    ) -> None:
        """
        幂等写入向量Point。

        Point ID固定使用Chunk ID。
        """

        if not records:
            return

        self.ensure_collection()

        points: list[models.PointStruct] = []

        for record in records:
            normalized_vector = self._normalize_vector(
                vector=record.vector,
                field_name="record vector",
            )

            self._validate_record(
                record=record,
                vector=normalized_vector,
            )

            points.append(
                models.PointStruct(
                    id=record.chunk_id,
                    vector=normalized_vector,
                    payload={
                        "chunk_id": record.chunk_id,
                        "document_id": (
                            record.document_id
                        ),
                        "chunk_index": (
                            record.chunk_index
                        ),
                        "filename": (
                            record.filename.strip()
                        ),
                        "content": record.content,
                        "embedding_model": (
                            record.embedding_model
                            .strip()
                        ),
                    },
                )
            )

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True,
        )

    def search(
        self,
        db: Session,
        query_vector: Sequence[float],
        embedding_model: str,
        top_k: int = 5,
        document_id: int | None = None,
    ) -> list[VectorSearchResult]:
        """
        在Qdrant中执行Dense Top-K检索。

        db参数用于兼容统一VectorStore接口，
        Qdrant实现本身不读取SQL。
        """

        del db

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

        normalized_query_vector = (
            self._normalize_vector(
                vector=query_vector,
                field_name="query vector",
            )
        )

        if (
            len(normalized_query_vector)
            != self.vector_size
        ):
            raise ValueError(
                "query vector dimension does not "
                "match collection dimension"
            )

        self.ensure_collection()

        response = self.client.query_points(
            collection_name=self.collection_name,
            query=normalized_query_vector,
            query_filter=self._build_filter(
                embedding_model=normalized_model,
                document_id=document_id,
            ),
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        )

        return [
            self._to_search_result(point)
            for point in response.points
        ]

    def delete_by_document_id(
        self,
        document_id: int,
    ) -> None:
        """
        删除指定文档的全部Qdrant Point。
        """

        if document_id <= 0:
            raise ValueError(
                "document_id must be greater than zero"
            )

        if not self.client.collection_exists(
            collection_name=self.collection_name,
        ):
            return

        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key=self.DOCUMENT_ID_FIELD,
                            match=models.MatchValue(
                                value=document_id
                            ),
                        )
                    ]
                )
            ),
            wait=True,
        )

    def _validate_collection(self) -> None:
        """
        校验已有Collection的向量配置。
        """

        collection = self.client.get_collection(
            collection_name=self.collection_name,
        )

        vectors_config = (
            collection.config.params.vectors
        )

        if not isinstance(
            vectors_config,
            models.VectorParams,
        ):
            raise RuntimeError(
                "named vectors are not supported"
            )

        if vectors_config.size != self.vector_size:
            raise RuntimeError(
                "Qdrant collection dimension "
                "does not match configured dimension"
            )

        if (
            vectors_config.distance
            != models.Distance.COSINE
        ):
            raise RuntimeError(
                "Qdrant collection distance "
                "must be cosine"
            )

    def _ensure_payload_indexes(self) -> None:
        """
        为常用过滤字段建立Payload索引。
        """

        collection = self.client.get_collection(
            collection_name=self.collection_name,
        )

        payload_schema = (
            collection.payload_schema or {}
        )

        if (
            self.DOCUMENT_ID_FIELD
            not in payload_schema
        ):
            self.client.create_payload_index(
                collection_name=(
                    self.collection_name
                ),
                field_name=(
                    self.DOCUMENT_ID_FIELD
                ),
                field_schema=(
                    models.PayloadSchemaType.INTEGER
                ),
                wait=True,
            )

        if (
            self.EMBEDDING_MODEL_FIELD
            not in payload_schema
        ):
            self.client.create_payload_index(
                collection_name=(
                    self.collection_name
                ),
                field_name=(
                    self.EMBEDDING_MODEL_FIELD
                ),
                field_schema=(
                    models.PayloadSchemaType.KEYWORD
                ),
                wait=True,
            )

    def _build_filter(
        self,
        embedding_model: str,
        document_id: int | None,
    ) -> models.Filter:
        """
        构造模型和文档过滤条件。
        """

        conditions = [
            models.FieldCondition(
                key=self.EMBEDDING_MODEL_FIELD,
                match=models.MatchValue(
                    value=embedding_model
                ),
            )
        ]

        if document_id is not None:
            conditions.append(
                models.FieldCondition(
                    key=self.DOCUMENT_ID_FIELD,
                    match=models.MatchValue(
                        value=document_id
                    ),
                )
            )

        return models.Filter(
            must=conditions
        )

    def _validate_record(
        self,
        record: VectorIndexRecord,
        vector: Sequence[float],
    ) -> None:
        """
        校验待写入的向量索引记录。
        """

        if record.chunk_id <= 0:
            raise ValueError(
                "chunk_id must be greater than zero"
            )

        if record.document_id <= 0:
            raise ValueError(
                "document_id must be greater than zero"
            )

        if record.chunk_index < 0:
            raise ValueError(
                "chunk_index cannot be negative"
            )

        if not record.filename.strip():
            raise ValueError(
                "filename cannot be empty"
            )

        if not record.content.strip():
            raise ValueError(
                "content cannot be empty"
            )

        if not record.embedding_model.strip():
            raise ValueError(
                "embedding_model cannot be empty"
            )

        if len(vector) != self.vector_size:
            raise ValueError(
                "record vector dimension does not "
                "match collection dimension"
            )

    @staticmethod
    def _normalize_vector(
        vector: Sequence[float],
        field_name: str,
    ) -> list[float]:
        """
        校验并规范化向量。
        """

        if not vector:
            raise ValueError(
                f"{field_name} cannot be empty"
            )

        normalized: list[float] = []

        for value in vector:
            if (
                isinstance(value, bool)
                or not isinstance(
                    value,
                    (int, float),
                )
            ):
                raise ValueError(
                    f"{field_name} must contain "
                    "only numeric values"
                )

            normalized_value = float(value)

            if not math.isfinite(
                normalized_value
            ):
                raise ValueError(
                    f"{field_name} contains "
                    "non-finite value"
                )

            normalized.append(
                normalized_value
            )

        return normalized

    @staticmethod
    def _to_search_result(
        point: models.ScoredPoint,
    ) -> VectorSearchResult:
        """
        将Qdrant Point转换为统一查询结果。
        """

        payload = point.payload

        if not isinstance(payload, dict):
            raise RuntimeError(
                "Qdrant point payload is missing"
            )

        required_fields = {
            "chunk_id",
            "document_id",
            "chunk_index",
            "filename",
            "content",
        }

        missing_fields = (
            required_fields - payload.keys()
        )

        if missing_fields:
            raise RuntimeError(
                "Qdrant point payload is incomplete: "
                f"{sorted(missing_fields)}"
            )

        filename = payload["filename"]
        content = payload["content"]

        if not isinstance(filename, str):
            raise RuntimeError(
                "Qdrant filename payload is invalid"
            )

        if not isinstance(content, str):
            raise RuntimeError(
                "Qdrant content payload is invalid"
            )

        return VectorSearchResult(
            document_id=int(
                payload["document_id"]
            ),
            filename=filename,
            chunk_id=int(
                payload["chunk_id"]
            ),
            chunk_index=int(
                payload["chunk_index"]
            ),
            content=content,
            score=float(point.score),
        )