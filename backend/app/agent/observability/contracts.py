"""Framework-neutral observability contracts for AgentOps.

Prompt bodies, model messages, tool arguments/results and retrieved document
contents are intentionally excluded from these contracts.
"""

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.constants.agent_runtime import AgentRuntime


class AgentObservationKind(str, Enum):
    """Stable component categories used by every Agent runtime."""

    AGENT_RUN = "agent_run"
    MODEL = "model"
    RETRIEVAL = "retrieval"
    TOOL = "tool"
    MCP = "mcp"
    GRAPH_NODE = "graph_node"


class AgentModelCallMode(str, Enum):
    """Stable model-call modes emitted by Agent runtimes."""

    TOOL_CALLING = "tool_calling"


class AgentGraphExecutionMode(str, Enum):
    """Whether a graph node belongs to a fresh run or durable resume."""

    FRESH = "fresh"
    RESUME = "resume"


class AgentTraceContext(BaseModel):
    """One Agent request's framework-neutral observability identity."""

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
        "trace_id", "request_id", "thread_id", "agent_version", "prompt_version",
        "toolset_version", "retrieval_config_version", mode="before",
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


class AgentModelUsage(BaseModel):
    """Provider-neutral token usage; cost is intentionally deferred to A5."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)


class AgentModelCallContext(BaseModel):
    """Safe metadata for one model generation inside an Agent trace."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    span: AgentSpanContext
    model_provider: str = Field(min_length=1, max_length=64)
    model_name: str = Field(min_length=1, max_length=128)
    prompt_id: str = Field(min_length=1, max_length=128)
    prompt_version: str = Field(min_length=1, max_length=64)
    mode: AgentModelCallMode
    turn: int | None = Field(default=None, ge=1)

    @field_validator("model_provider", "model_name", "prompt_id", "prompt_version", mode="before")
    @classmethod
    def normalize_text(cls, value: object) -> object:
        if isinstance(value, str):
            normalized = value.strip()
            return normalized or None
        return value


class AgentComponentCallContext(BaseModel):
    """Reviewed metadata for Retrieval/Tool/MCP/Graph child observations."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    span: AgentSpanContext
    turn: int | None = Field(default=None, ge=1)
    call_id: str | None = Field(default=None, min_length=1, max_length=128)
    tool_name: str | None = Field(default=None, min_length=1, max_length=128)
    tool_version: str | None = Field(default=None, min_length=1, max_length=64)
    tool_source: str | None = Field(default=None, min_length=1, max_length=32)
    mcp_server_id: str | None = Field(default=None, min_length=1, max_length=64)
    retrieval_mode: str | None = Field(default=None, min_length=1, max_length=32)
    top_k: int | None = Field(default=None, ge=1, le=1000)
    graph_node: str | None = Field(default=None, min_length=1, max_length=128)
    graph_execution_mode: AgentGraphExecutionMode | None = None

    @field_validator(
        "call_id", "tool_name", "tool_version", "tool_source", "mcp_server_id",
        "retrieval_mode", "graph_node", mode="before",
    )
    @classmethod
    def normalize_optional_text(cls, value: object) -> object:
        if isinstance(value, str):
            normalized = value.strip()
            return normalized or None
        return value

    @model_validator(mode="after")
    def validate_kind_metadata(self) -> "AgentComponentCallContext":
        kind = self.span.kind
        if kind is AgentObservationKind.TOOL and self.tool_name is None:
            raise ValueError("tool observation requires tool_name")
        if kind is AgentObservationKind.RETRIEVAL and self.retrieval_mode is None:
            raise ValueError("retrieval observation requires retrieval_mode")
        if kind is AgentObservationKind.MCP and (
            self.tool_name is None or self.mcp_server_id is None
        ):
            raise ValueError("mcp observation requires tool_name and mcp_server_id")
        if kind is AgentObservationKind.GRAPH_NODE and (
            self.graph_node is None or self.graph_execution_mode is None
        ):
            raise ValueError("graph node observation requires graph metadata")
        return self


class AgentComponentResult(BaseModel):
    """Safe result counters/state for a component observation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    result_count: int | None = Field(default=None, ge=0)
    evidence_count: int | None = Field(default=None, ge=0)
    next_route: str | None = Field(default=None, min_length=1, max_length=64)
    state_status: str | None = Field(default=None, min_length=1, max_length=64)

    @field_validator("next_route", "state_status", mode="before")
    @classmethod
    def normalize_optional_text(cls, value: object) -> object:
        if isinstance(value, str):
            normalized = value.strip()
            return normalized or None
        return value
