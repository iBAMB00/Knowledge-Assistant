"""Framework-neutral observability contracts for AgentOps.

Prompt bodies, model messages, tool arguments/results and retrieved document
contents are intentionally excluded from these contracts.
"""

from decimal import Decimal
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


class AgentRunOutcome(str, Enum):
    """Business outcome of one Agent run for observability dashboards."""

    COMPLETED = "completed"
    DEGRADED = "degraded"
    WAITING = "waiting"
    CANCELLED = "cancelled"
    FAILED = "failed"


class AgentErrorStage(str, Enum):
    """Stable stage labels used by structured Agent observability errors."""

    AGENT = "agent"
    MODEL = "model"
    TOOL = "tool"
    RETRIEVAL = "retrieval"
    MCP = "mcp"
    GRAPH_NODE = "graph_node"
    STREAM = "stream"


class AgentObservationError(BaseModel):
    """Safe, bounded error facts that may be exported to an observability vendor.

    Raw traceback, request/response bodies, credentials and document contents are
    deliberately excluded. ``safe_message`` must already be redacted/bounded.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    error_code: str = Field(min_length=1, max_length=128)
    error_type: str = Field(min_length=1, max_length=128)
    stage: AgentErrorStage
    substage: str | None = Field(default=None, min_length=1, max_length=64)
    safe_message: str | None = Field(default=None, min_length=1, max_length=320)
    root_cause_type: str | None = Field(default=None, min_length=1, max_length=128)
    provider: str | None = Field(default=None, min_length=1, max_length=64)
    model: str | None = Field(default=None, min_length=1, max_length=128)
    provider_error_code: str | None = Field(default=None, min_length=1, max_length=128)
    http_status: int | None = Field(default=None, ge=100, le=599)
    retryable: bool | None = None
    fail_open: bool | None = None
    fingerprint: str = Field(min_length=8, max_length=64)

    @field_validator(
        "error_code", "error_type", "substage", "safe_message",
        "root_cause_type", "provider", "model", "provider_error_code", "fingerprint", mode="before",
    )
    @classmethod
    def normalize_error_text(cls, value: object) -> object:
        if isinstance(value, str):
            normalized = value.strip()
            return normalized or None
        return value


class AgentModelCost(BaseModel):
    """Provider-neutral USD cost buckets for one model generation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    input_cost_usd: Decimal | None = Field(default=None, ge=0)
    output_cost_usd: Decimal | None = Field(default=None, ge=0)
    total_cost_usd: Decimal | None = Field(default=None, ge=0)


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
    input_preview: str | None = Field(default=None, min_length=1, max_length=500)
    agent_version: str = Field(min_length=1, max_length=64)
    prompt_id: str = Field(min_length=1, max_length=128)
    prompt_version: str = Field(min_length=1, max_length=64)
    toolset_version: str = Field(min_length=1, max_length=64)
    retrieval_config_version: str = Field(min_length=1, max_length=64)

    @field_validator(
        "trace_id", "request_id", "thread_id", "input_preview", "agent_version", "prompt_id", "prompt_version",
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
    """Provider-neutral token usage facts used by A5 cost estimation."""

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




class AgentRunMetrics(BaseModel):
    """Provider-neutral run summary derived from traced child operations."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    success: bool
    outcome: AgentRunOutcome
    error_type: str | None = Field(default=None, min_length=1, max_length=128)
    run_latency_ms: float = Field(ge=0)

    model_calls: int = Field(default=0, ge=0)
    tool_calls: int = Field(default=0, ge=0)
    retrieval_calls: int = Field(default=0, ge=0)
    mcp_calls: int = Field(default=0, ge=0)
    graph_node_calls: int = Field(default=0, ge=0)
    failed_calls: int = Field(default=0, ge=0)
    warning_calls: int = Field(default=0, ge=0)

    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)
    model_usage_missing_calls: int = Field(default=0, ge=0)

    model_latency_ms: float = Field(default=0, ge=0)
    tool_latency_ms: float = Field(default=0, ge=0)
    retrieval_latency_ms: float = Field(default=0, ge=0)
    mcp_latency_ms: float = Field(default=0, ge=0)
    graph_node_latency_ms: float = Field(default=0, ge=0)

    estimated_cost_usd: Decimal | None = Field(default=None, ge=0)
    pricing_configured: bool = False
    cost_estimate_complete: bool = False
    pricing_version: str | None = Field(default=None, min_length=1, max_length=64)
    first_error: AgentObservationError | None = None
    first_warning: AgentObservationError | None = None

    @model_validator(mode="before")
    @classmethod
    def derive_outcome_when_omitted(cls, data: object) -> object:
        if not isinstance(data, dict) or data.get("outcome") is not None:
            return data
        normalized = dict(data)
        success = bool(normalized.get("success"))
        failed_calls = normalized.get("failed_calls", 0)
        warning_calls = normalized.get("warning_calls", 0)
        normalized["outcome"] = (
            AgentRunOutcome.FAILED
            if not success
            else (
                AgentRunOutcome.DEGRADED
                if (
                    isinstance(failed_calls, int) and failed_calls > 0
                ) or (
                    isinstance(warning_calls, int) and warning_calls > 0
                )
                else AgentRunOutcome.COMPLETED
            )
        )
        return normalized

    @field_validator("error_type", "pricing_version", mode="before")
    @classmethod
    def normalize_optional_text(cls, value: object) -> object:
        if isinstance(value, str):
            normalized = value.strip()
            return normalized or None
        return value

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
