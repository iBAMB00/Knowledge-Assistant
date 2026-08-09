from pydantic import BaseModel, Field


class KnowledgeChatSource(BaseModel):
    """知识库问答对外来源。"""

    source_number: int = Field(gt=0)
    document_id: int = Field(gt=0)
    filename: str = Field(min_length=1)
    chunk_id: int = Field(gt=0)
    excerpt: str = Field(min_length=1)
    section_title: str | None = None
    heading_path: list[str] = Field(default_factory=list)
    start_page: int | None = Field(default=None, gt=0)
    end_page: int | None = Field(default=None, gt=0)
    page_numbers: list[int] = Field(default_factory=list)


class KnowledgeChatResponse(BaseModel):
    """知识库问答结果。"""

    answer: str
    sources: list[KnowledgeChatSource]
