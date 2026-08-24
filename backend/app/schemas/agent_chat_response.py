from pydantic import BaseModel, ConfigDict, Field

from app.agent.context_engine import AgentContextUsage


class AgentChatResponse(BaseModel):
    """同步 Agent 对外最小响应。"""

    model_config = ConfigDict(extra="forbid")

    answer: str = Field(min_length=1)
    context_usage: AgentContextUsage | None = None
