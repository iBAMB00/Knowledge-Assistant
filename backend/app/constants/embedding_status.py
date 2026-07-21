from enum import Enum


class EmbeddingStatus(str, Enum):
    """
    Chunk向量化状态。
    """

    PENDING = "pending"

    PROCESSING = "processing"

    COMPLETED = "completed"

    FAILED = "failed"