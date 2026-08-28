"""Provider-neutral observability boundary for v2.5 AgentOps."""

from typing import Protocol, runtime_checkable

from app.agent.observability.contracts import (
    AgentComponentCallContext,
    AgentComponentResult,
    AgentModelCallContext,
    AgentModelCost,
    AgentObservationError,
    AgentModelUsage,
    AgentRunMetrics,
    AgentTraceContext,
)


@runtime_checkable
class AgentComponentCallHandle(Protocol):
    @property
    def span_id(self) -> str: ...

    def finish(
        self,
        *,
        ok: bool = True,
        result: AgentComponentResult | None = None,
        error_code: str | None = None,
        error: AgentObservationError | None = None,
        warning: AgentObservationError | None = None,
    ) -> None: ...


@runtime_checkable
class AgentModelCallHandle(Protocol):
    @property
    def span_id(self) -> str: ...

    def finish(
        self,
        *,
        ok: bool = True,
        usage: AgentModelUsage | None = None,
        cost: AgentModelCost | None = None,
        error_code: str | None = None,
        error: AgentObservationError | None = None,
    ) -> None: ...


@runtime_checkable
class AgentTraceHandle(Protocol):
    @property
    def trace_id(self) -> str: ...

    @property
    def provider_trace_id(self) -> str | None: ...

    def start_model_call(
        self,
        *,
        call_context: AgentModelCallContext,
    ) -> AgentModelCallHandle: ...

    def start_component_call(
        self,
        *,
        call_context: AgentComponentCallContext,
    ) -> AgentComponentCallHandle: ...

    def finish(
        self,
        *,
        ok: bool = True,
        error_code: str | None = None,
        metrics: AgentRunMetrics | None = None,
        error: AgentObservationError | None = None,
    ) -> None: ...


@runtime_checkable
class ObservabilityProvider(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def enabled(self) -> bool: ...

    def start_trace(
        self,
        *,
        trace_context: AgentTraceContext,
        name: str = "agent.chat",
    ) -> AgentTraceHandle: ...

    def publish_trace_score(
        self,
        *,
        provider_trace_id: str,
        name: str,
        value: float,
        data_type: str = "NUMERIC",
        comment: str | None = None,
        score_id: str | None = None,
    ) -> bool: ...

    def flush(self) -> None: ...

    def shutdown(self) -> None: ...
