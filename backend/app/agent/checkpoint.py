"""Framework-neutral durable checkpoint and recovery contracts."""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator
from sqlalchemy.orm import Session

from app.agent.model_response import (
    LLMToolCall,
    LLMToolExchange,
    LLMToolResponse,
)
from app.agent.run_event import AgentToolResultEvent
from app.agent.state import AGENT_STATE_SCHEMA_VERSION, AgentState
from app.constants.agent_state_status import AgentStateStatus


CHECKPOINT_SCHEMA_VERSION = "1.0"


class AgentExecutionCheckpointPayload(BaseModel):
    """可序列化、可落库的 Graph 执行快照。"""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    checkpoint_schema_version: str = CHECKPOINT_SCHEMA_VERSION
    agent_state: AgentState
    history: tuple[LLMToolExchange, ...] = ()
    pending_tool_calls: tuple[LLMToolCall, ...] = ()
    last_model_response: LLMToolResponse | None = None
    tool_observations: tuple[AgentToolResultEvent, ...] = ()
    final_answer: str | None = None
    turn: int = Field(default=0, ge=0)
    tool_call_count: int = Field(default=0, ge=0)
    seen_tool_call_signatures: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_checkpoint_contract(self) -> "AgentExecutionCheckpointPayload":
        if self.checkpoint_schema_version != CHECKPOINT_SCHEMA_VERSION:
            raise ValueError("unsupported checkpoint schema version")
        if self.agent_state.state_schema_version != AGENT_STATE_SCHEMA_VERSION:
            raise ValueError("unsupported agent state schema version")

        if self.pending_tool_calls:
            if self.last_model_response is None:
                raise ValueError(
                    "pending tool calls require last model response"
                )
            if tuple(self.last_model_response.tool_calls) != self.pending_tool_calls:
                raise ValueError(
                    "pending tool calls do not match last model response"
                )

        if self.final_answer is not None:
            if self.agent_state.status is not AgentStateStatus.SUCCEEDED:
                raise ValueError(
                    "final answer requires succeeded agent state"
                )
            if not self.final_answer.strip():
                raise ValueError("final answer cannot be empty")

        return self


class AgentCheckpointWriter(Protocol):
    """Runner 只依赖这一最小写入边界，不依赖 SQLAlchemy Repository。"""

    def save_checkpoint(
        self,
        db: Session,
        payload: AgentExecutionCheckpointPayload,
    ) -> object:
        ...


class AgentRecoveryLoader(Protocol):
    """Runner 读取可恢复 checkpoint 的最小边界。"""

    def load_resume_checkpoint(
        self,
        db: Session,
        *,
        thread_id: str,
        user_id: int,
        knowledge_base_id: int,
    ) -> AgentExecutionCheckpointPayload:
        ...


class AgentResumeError(RuntimeError):
    """Stateful Agent 恢复执行失败。"""


class AgentResumeCheckpointNotFoundError(AgentResumeError):
    """当前可信 Scope 下不存在可读取的 checkpoint。"""


class AgentResumeStateError(AgentResumeError):
    """Checkpoint 存在，但当前状态不允许恢复执行。"""
