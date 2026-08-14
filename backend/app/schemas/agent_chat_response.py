from pydantic import BaseModel, ConfigDict, Field


class AgentChatResponse(BaseModel):
    """Native Agent 对外最小响应。"""

    model_config = ConfigDict(extra="forbid")

    answer: str = Field(min_length=1)
