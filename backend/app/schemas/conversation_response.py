from datetime import datetime

from pydantic import BaseModel, ConfigDict

from app.constants.conversation_message_role import ConversationMessageRole
from app.constants.conversation_mode import ConversationMode


class ConversationResponse(BaseModel):
    """用户自己的 Conversation 列表 / 详情安全视图。"""

    model_config = ConfigDict(from_attributes=True)

    id: int
    mode: ConversationMode
    knowledge_base_id: int
    title: str | None
    created_at: datetime
    updated_at: datetime


class ConversationMessageResponse(BaseModel):
    """用户可见 Conversation Message。"""

    model_config = ConfigDict(from_attributes=True)

    id: int
    role: ConversationMessageRole
    content: str
    created_at: datetime
