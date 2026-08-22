from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.constants.conversation_mode import ConversationMode


class ConversationCreateRequest(BaseModel):
    """创建一个固定 mode / knowledge_base 范围的用户对话。"""

    model_config = ConfigDict(extra="forbid")

    mode: ConversationMode
    knowledge_base_id: int = Field(strict=True, gt=0)
    title: str | None = Field(default=None, max_length=200)

    @field_validator("title", mode="before")
    @classmethod
    def normalize_title(cls, value: object) -> object:
        if isinstance(value, str):
            normalized = value.strip()
            return normalized or None
        return value
