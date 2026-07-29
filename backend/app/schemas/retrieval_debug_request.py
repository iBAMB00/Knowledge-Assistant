from pydantic import BaseModel, Field

class RetrievalDebugRequest(BaseModel):
    """
    检索调试请求。
    """

    query: str

    top_k: int | None = Field(
        default=None,
        gt=0,
    )

    candidate_k: int | None = Field(
        default=None,
        gt=0,
    )

    score_threshold: float | None = Field(
        default=None,
        ge=-1.0,
        le=1.0,
    )

    per_document_limit: int | None = Field(
        default=None,
        gt=0,
    )

    document_id: int | None = Field(
        default=None,
        gt=0,
    )