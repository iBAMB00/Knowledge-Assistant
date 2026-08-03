from pydantic import BaseModel, Field


class VectorSearchResult(BaseModel):
    """
    向量检索结果。
    """

    document_id: int

    filename: str = Field(min_length=1)

    chunk_id: int

    chunk_index: int

    content: str

    score: float