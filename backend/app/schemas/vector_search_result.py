from pydantic import BaseModel


class VectorSearchResult(BaseModel):
    """
    向量检索结果。
    """

    document_id: int

    chunk_id: int

    chunk_index: int

    content: str

    score: float