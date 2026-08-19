from typing import Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field


class AgentStatusEvent(BaseModel):
    """Agent 进入一次模型决策阶段的安全运行事件。"""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    type: Literal["status"] = "status"
    stage: Literal["model"] = "model"
    turn: int = Field(ge=1)


class AgentToolCallEvent(BaseModel):
    """Agent 准备执行 Tool 的安全运行事件，不暴露 Tool 参数正文。"""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    type: Literal["tool_call"] = "tool_call"
    turn: int = Field(ge=1)
    call_id: str = Field(min_length=1)
    tool_name: str = Field(min_length=1)


class AgentToolResultEvent(BaseModel):
    """Tool 执行完成事件，只暴露成功状态与安全错误码。"""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    type: Literal["tool_result"] = "tool_result"
    turn: int = Field(ge=1)
    call_id: str = Field(min_length=1)
    tool_name: str = Field(min_length=1)
    ok: bool
    duration_ms: int = Field(ge=0)
    error_code: str | None = None


class AgentMessageEvent(BaseModel):
    """Agent 最终回答事件。当前 B5 不伪装成 token streaming。"""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    type: Literal["message"] = "message"
    content: str = Field(min_length=1)
    turns: int = Field(ge=1)
    tool_call_count: int = Field(ge=0)


AgentRunEvent: TypeAlias = (
    AgentStatusEvent
    | AgentToolCallEvent
    | AgentToolResultEvent
    | AgentMessageEvent
)
