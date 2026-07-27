from pydantic import BaseModel, Field


class RetrievalDebugRequest(BaseModel):
    """
    检索调试请求。
    """

    query: str

    top_k: int = Field(
        default=5,
        gt=0,
    )

    candidate_k: int = Field(
        default=20,
        gt=0,
    )

    score_threshold: float = Field(
        default=-1.0,
        ge=-1.0,
        le=1.0,
    )

    per_document_limit: int = Field(
        default=2,
        gt=0,
    )

    document_id: int | None = Field(
        default=None,
        gt=0,
    )