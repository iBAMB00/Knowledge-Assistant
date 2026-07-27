from pydantic import BaseModel, Field


class KnowledgeChatRequest(BaseModel):
    """
    知识库问答请求。
    """

    question: str

    top_k: int = Field(
        default=5,
        gt=0,
    )

    score_threshold: float = Field(
        default=-1.0,
        ge=-1.0,
        le=1.0,
    )

    document_id: int | None = Field(
        default=None,
        gt=0,
    )