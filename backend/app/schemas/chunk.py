from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ChunkResult:
    """
    单个文本切片的算法输出结果。

    这是切片算法与上层服务之间的数据契约，
    不属于数据库模型，也不属于 API 请求响应模型。
    """

    content: str
    chunk_index: int
    start_offset: int
    end_offset: int

    token_count: int | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict,
    )

    parent_chunk_index: int | None = None  # 预留：父切片索引，用于递归切片


    def __post_init__(self) -> None:
        """
        保证切片结果在创建时就是有效状态。
        """

        if not self.content.strip():
            raise ValueError(
                "Chunk content cannot be empty.",
            )

        if self.chunk_index < 0:
            raise ValueError(
                "Chunk index cannot be negative.",
            )

        if self.start_offset < 0:
            raise ValueError(
                "Chunk start offset cannot be negative.",
            )

        if self.end_offset <= self.start_offset:
            raise ValueError(
                "Chunk end offset must be greater than start offset.",
            )

        if self.token_count is not None and self.token_count < 0:
            raise ValueError(
                "Chunk token count cannot be negative.",
            )

        if (
            self.parent_chunk_index is not None
            and self.parent_chunk_index < 0
        ):
            raise ValueError(
                "Parent chunk index cannot be negative.",
            )