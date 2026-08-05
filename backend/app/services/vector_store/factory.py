from dataclasses import dataclass
from functools import lru_cache

from qdrant_client import QdrantClient

from app.core.config import Settings, get_settings
from app.repositories.chunk_embedding_repository import ChunkEmbeddingRepository
from app.services.vector_store.base import VectorIndex, VectorStore
from app.services.vector_store.database import DatabaseVectorStore
from app.services.vector_store.qdrant import QdrantVectorStore


@dataclass(frozen=True)
class VectorStoreComponents:
    """向量检索与外部索引组件。"""

    vector_store: VectorStore
    vector_index: VectorIndex | None


class VectorStoreFactory:
    """根据应用配置创建向量存储组件。"""

    @staticmethod
    def create(
        settings: Settings | None = None,
        qdrant_client: QdrantClient | None = None,
    ) -> VectorStoreComponents:
        """创建当前配置对应的向量存储组件。"""

        resolved_settings = settings or get_settings()

        if resolved_settings.vector_store_backend == "database":
            vector_store = DatabaseVectorStore(
                chunk_embedding_repository=ChunkEmbeddingRepository()
            )
            return VectorStoreComponents(vector_store=vector_store, vector_index=None)

        if resolved_settings.vector_store_backend == "qdrant":
            client = qdrant_client or QdrantClient(
                url=resolved_settings.qdrant_url,
                api_key=resolved_settings.qdrant_api_key,
                timeout=resolved_settings.qdrant_timeout,
            )

            vector_store = QdrantVectorStore(
                client=client,
                collection_name=resolved_settings.qdrant_collection_name,
                vector_size=resolved_settings.embedding_dimension,
            )

            return VectorStoreComponents(
                vector_store=vector_store,
                vector_index=vector_store,
            )

        raise ValueError(
            f"unsupported vector store backend: "
            f"{resolved_settings.vector_store_backend}"
        )


@lru_cache
def get_vector_store_components() -> VectorStoreComponents:
    """获取当前进程共享的向量存储组件。"""

    return VectorStoreFactory.create()