from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from app.constants.agent_state_status import AgentStateStatus
from app.constants.conversation_mode import ConversationMode
from app.schemas.conversation_contract import (
    ConversationMessagePayload,
    ConversationScope,
)


AGENT_STATE_SCHEMA_VERSION = "1.0"


class AgentThreadIdentity(BaseModel):
    """
    Stateful Runtime 的连续执行线程标识。

    Thread 是 Runtime 概念，不等于 Conversation，也不等于 AgentRun：
    - Conversation：用户看到的一段聊天；
    - Thread：同一 Agent Conversation 的连续状态上下文；
    - AgentRun：Thread 中一次具体执行事实。
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    thread_id: str = Field(min_length=1, max_length=128)
    conversation_id: int = Field(strict=True, gt=0)

    @field_validator("thread_id", mode="before")
    @classmethod
    def normalize_thread_id(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip()
        return value


class AgentState(BaseModel):
    """
    v2.3 Stateful Runtime 的框架无关状态 Contract。

    设计约束：
    - 只保存可序列化 raw data，不保存拼装后的 Prompt；
    - 不依赖 LangGraph / LangChain 类型；
    - Conversation Scope 在 Thread 生命周期内保持固定；
    - 当前只建立最小稳定字段，plan / evidence / tool_results /
      pending_action 会在对应小版本出现真实消费方后再扩展。
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    state_schema_version: Literal["1.0"] = AGENT_STATE_SCHEMA_VERSION
    conversation: ConversationScope
    thread: AgentThreadIdentity
    agent_run_id: int | str | None = None
    status: AgentStateStatus = AgentStateStatus.READY
    messages: tuple[ConversationMessagePayload, ...] = ()
    task: str | None = Field(default=None, min_length=1, max_length=20000)
    retry_count: int = Field(default=0, strict=True, ge=0)
    last_error_code: str | None = Field(default=None, max_length=100)

    @field_validator("task", "last_error_code", mode="before")
    @classmethod
    def normalize_optional_text(cls, value: object) -> object:
        if isinstance(value, str):
            normalized = value.strip()
            return normalized or None
        return value

    @model_validator(mode="after")
    def validate_identity_boundary(self) -> "AgentState":
        if self.conversation.mode != ConversationMode.AGENT:
            raise ValueError(
                "AgentState requires an agent-mode conversation"
            )

        if self.thread.conversation_id != self.conversation.conversation_id:
            raise ValueError(
                "thread conversation_id must match conversation scope"
            )

        return self
