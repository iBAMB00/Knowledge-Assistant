from datetime import datetime

from pydantic import BaseModel


class ChunkResponse(BaseModel):
    """
    文档切片响应模型。
    """

    id: int

    chunk_index: int

    content: str

    token_count: int | None

    chunk_strategy: str

    created_at: datetime