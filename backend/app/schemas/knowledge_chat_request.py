from pydantic import BaseModel, Field


class KnowledgeChatRequest(BaseModel):
    """
    知识库问答请求。
    """

    question: str

    top_k: int | None = Field(
        default=None,
        gt=0,
    )

    document_id: int | None = Field(
        default=None,
        gt=0,
    )