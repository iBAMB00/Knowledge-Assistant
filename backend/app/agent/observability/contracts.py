"""Framework-neutral observability contracts for AgentOps.

v2.5-A1 intentionally defines only safe execution metadata. Prompt bodies,
model messages, tool arguments, tool results and retrieved document contents
are not part of these contracts.
"""

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.constants.agent_runtime import AgentRuntime


class AgentObservationKind(str, Enum):
    """Stable component categories used by every Agent runtime."""

    AGENT_RUN = "agent_run"
    MODEL = "model"
    RETRIEVAL = "retrieval"
    TOOL = "tool"
    MCP = "mcp"
    GRAPH_NODE = "graph_node"


class AgentTraceContext(BaseModel):
    """One Agent request's framework-neutral observability identity.

    ``AgentRun`` remains the persisted business execution fact. A Trace is an
    observability view that may start before an AgentRun database row exists,
    so ``agent_run_id`` is deliberately optional and can be bound later.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    trace_id: str = Field(min_length=1, max_length=128)
    request_id: str = Field(min_length=1, max_length=128)
    runtime: AgentRuntime
    user_id: int = Field(strict=True, gt=0)
    knowledge_base_id: int = Field(strict=True, gt=0)
    conversation_id: int | None = Field(default=None, strict=True, gt=0)
    thread_id: str | None = Field(default=None, min_length=1, max_length=128)
    agent_run_id: int | str | None = None
    agent_version: str = Field(min_length=1, max_length=64)
    prompt_version: str = Field(min_length=1, max_length=64)
    toolset_version: str = Field(min_length=1, max_length=64)
    retrieval_config_version: str = Field(min_length=1, max_length=64)

    @field_validator(
        "trace_id",
        "request_id",
        "thread_id",
        "agent_version",
        "prompt_version",
        "toolset_version",
        "retrieval_config_version",
        mode="before",
    )
    @classmethod
    def normalize_text(cls, value: object) -> object:
        if isinstance(value, str):
            normalized = value.strip()
            return normalized or None
        return value

    @field_validator("agent_run_id", mode="before")
    @classmethod
    def normalize_agent_run_id(cls, value: object) -> object:
        if isinstance(value, str):
            normalized = value.strip()
            if not normalized:
                raise ValueError("agent_run_id cannot be empty")
            return normalized
        return value


class AgentSpanContext(BaseModel):
    """Safe identity for one child operation inside an Agent Trace."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    trace_id: str = Field(min_length=1, max_length=128)
    span_id: str = Field(min_length=1, max_length=128)
    parent_span_id: str | None = Field(default=None, min_length=1, max_length=128)
    kind: AgentObservationKind
    name: str = Field(min_length=1, max_length=128)

    @field_validator("trace_id", "span_id", "parent_span_id", "name", mode="before")
    @classmethod
    def normalize_text(cls, value: object) -> object:
        if isinstance(value, str):
            normalized = value.strip()
            return normalized or None
        return value
