from pydantic import BaseModel


class ChunkSummaryResponse(BaseModel):
    """
    文档切片统计响应。
    """

    document_id: int

    chunk_count: int

    total_characters: int