from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.constants.conversation_message_role import ConversationMessageRole
from app.constants.conversation_mode import ConversationMode


class ConversationScope(BaseModel):
    """
    Conversation 的稳定身份与数据隔离范围。

    v2.3 第一版冻结规则：
    - Conversation 属于 User；
    - 一个 Conversation 固定绑定一个 mode；
    - 一个 Conversation 固定绑定一个 knowledge_base_id；
    - title / created_at 等属于后续 Persistence / UI 元数据，
      不参与本 Contract 的安全范围定义。
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    conversation_id: int = Field(strict=True, gt=0)
    user_id: int = Field(strict=True, gt=0)
    mode: ConversationMode
    knowledge_base_id: int = Field(strict=True, gt=0)


class ConversationMessagePayload(BaseModel):
    """
    Conversation 中用户可见消息的框架无关载荷。

    这里只保存 user / assistant 正文。Tool 调用、Tool Result、
    System Prompt 和隐藏推理继续留在 Agent Runtime / Lifecycle 边界内。
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    role: ConversationMessageRole
    content: str = Field(min_length=1, max_length=20000)

    @field_validator("content", mode="before")
    @classmethod
    def normalize_content(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip()
        return value
