from datetime import datetime

from pydantic import BaseModel, ConfigDict

from app.constants.agent_run_status import AgentRunStatus
from app.constants.agent_tool_call_status import AgentToolCallStatus


class AgentRunSummaryResponse(BaseModel):
    """AgentRun 对外列表摘要，不暴露 user_id 等内部身份字段。"""

    model_config = ConfigDict(from_attributes=True)

    id: int
    knowledge_base_id: int
    status: AgentRunStatus
    tool_call_count: int
    started_at: datetime
    finished_at: datetime | None


class AgentToolCallSummaryResponse(BaseModel):
    """单次 Tool 调用的安全运行摘要。"""

    model_config = ConfigDict(from_attributes=True)

    tool_name: str
    tool_version: str
    status: AgentToolCallStatus
    duration_ms: int | None
    error_type: str | None
    started_at: datetime
    finished_at: datetime | None


class AgentRunDetailResponse(AgentRunSummaryResponse):
    """AgentRun 详情，仅包含可观测运行事实和 Tool 摘要。"""

    request_id: str
    model_provider: str
    model_name: str
    error_type: str | None
    tool_calls: list[AgentToolCallSummaryResponse]
