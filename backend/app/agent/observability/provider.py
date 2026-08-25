"""Provider-neutral observability boundary for v2.5 AgentOps."""

from typing import Protocol, runtime_checkable

from app.agent.observability.contracts import (
    AgentComponentCallContext,
    AgentComponentResult,
    AgentModelCallContext,
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
        error_code: str | None = None,
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
        name: str = "agent.run",
    ) -> AgentTraceHandle: ...

    def flush(self) -> None: ...

    def shutdown(self) -> None: ...
