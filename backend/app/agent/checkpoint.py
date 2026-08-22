"""Framework-neutral durable checkpoint contract for Stateful Agent execution."""

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
    def validate_schema_versions(self) -> "AgentExecutionCheckpointPayload":
        if self.checkpoint_schema_version != CHECKPOINT_SCHEMA_VERSION:
            raise ValueError("unsupported checkpoint schema version")
        if self.agent_state.state_schema_version != AGENT_STATE_SCHEMA_VERSION:
            raise ValueError("unsupported agent state schema version")
        return self


class AgentCheckpointWriter(Protocol):
    """Runner 只依赖这一最小写入边界，不依赖 SQLAlchemy Repository。"""

    def save_checkpoint(
        self,
        db: Session,
        payload: AgentExecutionCheckpointPayload,
    ) -> object:
        ...
