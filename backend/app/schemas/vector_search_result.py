from pydantic import BaseModel, Field


class VectorSearchResult(BaseModel):
    """
    向量检索结果。

    Parent-Child模式下chunk_id仍表示实际命中的Child，
    content会在检索编排层扩展为Parent正文，便于同时保留
    细粒度命中证据和较完整的LLM上下文。
    """

    document_id: int

    filename: str = Field(min_length=1)

    chunk_id: int

    chunk_index: int

    content: str

    score: float

    parent_chunk_id: int | None = Field(
        default=None,
        exclude=True,
    )
