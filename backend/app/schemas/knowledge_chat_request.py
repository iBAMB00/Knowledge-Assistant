from pydantic import BaseModel, Field, ConfigDict


class KnowledgeChatRequest(BaseModel):
    """
    知识库问答请求。
    """

    model_config = ConfigDict(extra="forbid")

    question: str

    knowledge_base_id: int = Field(gt=0)

    conversation_id: int | None = Field(
        default=None,
        gt=0,
    )

    top_k: int | None = Field(
        default=None,
        gt=0,
    )

    document_id: int | None = Field(
        default=None,
        gt=0,
    )