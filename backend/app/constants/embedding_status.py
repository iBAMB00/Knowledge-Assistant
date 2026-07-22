from enum import Enum


class EmbeddingStatus(str, Enum):
    """
    单个DocumentChunk的向量化状态。
    """

    # 等待向量化
    PENDING = "pending"

    # 正在生成向量
    PROCESSING = "processing"

    # 向量已经生成并保存
    COMPLETED = "completed"

    # 向量生成失败，可以重试
    FAILED = "failed"