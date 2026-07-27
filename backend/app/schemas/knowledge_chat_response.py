from pydantic import BaseModel

from app.schemas.vector_search_result import (
    VectorSearchResult,
)


class KnowledgeChatResponse(BaseModel):
    """
    知识库问答结果。
    """

    answer: str

    sources: list[VectorSearchResult]