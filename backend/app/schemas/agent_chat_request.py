from pydantic import BaseModel, ConfigDict, Field


class AgentChatRequest(BaseModel):
    """Native Agent 问答请求。"""

    model_config = ConfigDict(extra="forbid")

    message: str
    knowledge_base_id: int = Field(gt=0)
