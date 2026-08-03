from pydantic import BaseModel, Field


class KnowledgeChatSource(BaseModel):
    """知识库问答对外来源。"""

    source_number: int = Field(gt=0)
    document_id: int = Field(gt=0)
    filename: str = Field(min_length=1)
    excerpt: str = Field(min_length=1)


class KnowledgeChatResponse(BaseModel):
    """知识库问答结果。"""

    answer: str
    sources: list[KnowledgeChatSource]