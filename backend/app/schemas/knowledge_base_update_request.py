from pydantic import BaseModel, ConfigDict, Field, model_validator


class KnowledgeBaseUpdateRequest(BaseModel):
    """更新知识库请求。"""

    model_config = ConfigDict(extra="forbid")

    name: str | None = Field(default=None, min_length=1, max_length=100)
    description: str | None = Field(default=None, max_length=1000)

    @model_validator(mode="after")
    def require_changed_field(self) -> "KnowledgeBaseUpdateRequest":
        if not self.model_fields_set:
            raise ValueError("at least one field is required")
        return self
