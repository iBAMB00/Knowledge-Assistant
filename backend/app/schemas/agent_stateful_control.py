from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.constants.agent_state_status import AgentStateStatus


class AgentApprovalRequirementResponse(BaseModel):
    """对外只暴露人工确认所需的最小安全元数据。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    call_id: str = Field(min_length=1)
    tool_name: str = Field(min_length=1)
    reason: str = Field(min_length=1, max_length=500)


class AgentWaitingResponse(BaseModel):
    """同步 Stateful Chat 正常暂停时的 202 响应。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal["waiting"] = "waiting"
    thread_id: str = Field(min_length=1, max_length=128)
    approvals: list[AgentApprovalRequirementResponse]


class AgentThreadActionRequest(BaseModel):
    """Thread 控制操作的可信 KB Scope 输入。"""

    model_config = ConfigDict(extra="forbid")

    knowledge_base_id: int = Field(gt=0)


class AgentThreadApprovalRequest(AgentThreadActionRequest):
    """批准/拒绝待确认 ToolCall；空列表表示选择全部 pending approval。"""

    call_ids: list[str] = Field(default_factory=list)

    @field_validator("call_ids")
    @classmethod
    def validate_call_ids(cls, value: list[str]) -> list[str]:
        normalized = [item.strip() for item in value]
        if any(not item for item in normalized):
            raise ValueError("approval call_id cannot be empty")
        if len(set(normalized)) != len(normalized):
            raise ValueError("approval call_ids must be unique")
        return normalized


class AgentThreadStatusResponse(BaseModel):
    """Thread 对外运行状态；不暴露 checkpoint payload / Tool 参数 / Result 正文。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    thread_id: str = Field(min_length=1, max_length=128)
    conversation_id: int = Field(gt=0)
    knowledge_base_id: int = Field(gt=0)
    status: AgentStateStatus
    retry_count: int = Field(ge=0)
    last_error_code: str | None = None
    pending_approvals: list[AgentApprovalRequirementResponse]
    can_resume: bool
    can_approve: bool
    can_reject: bool
    can_cancel: bool
