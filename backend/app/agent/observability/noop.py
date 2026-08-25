"""No-op observability implementation used when tracing is disabled/degraded."""

from dataclasses import dataclass

from app.agent.observability.contracts import (
    AgentComponentCallContext,
    AgentComponentResult,
    AgentModelCallContext,
    AgentModelUsage,
    AgentTraceContext,
)


@dataclass(frozen=True, slots=True)
class NoOpComponentCallHandle:
    span_id: str

    def finish(self, *, ok: bool = True, result: AgentComponentResult | None = None, error_code: str | None = None) -> None:
        del ok, result, error_code


@dataclass(frozen=True, slots=True)
class NoOpModelCallHandle:
    span_id: str

    def finish(self, *, ok: bool = True, usage: AgentModelUsage | None = None, error_code: str | None = None) -> None:
        del ok, usage, error_code


@dataclass(frozen=True, slots=True)
class NoOpTraceHandle:
    trace_id: str

    @property
    def provider_trace_id(self) -> None:
        return None

    def start_model_call(self, *, call_context: AgentModelCallContext) -> NoOpModelCallHandle:
        return NoOpModelCallHandle(span_id=call_context.span.span_id)

    def start_component_call(self, *, call_context: AgentComponentCallContext) -> NoOpComponentCallHandle:
        return NoOpComponentCallHandle(span_id=call_context.span.span_id)

    def finish(self, *, ok: bool = True, error_code: str | None = None) -> None:
        del ok, error_code


class NoOpObservabilityProvider:
    name = "noop"
    enabled = False

    def start_trace(self, *, trace_context: AgentTraceContext, name: str = "agent.run") -> NoOpTraceHandle:
        del name
        return NoOpTraceHandle(trace_id=trace_context.trace_id)

    def flush(self) -> None:
        return None

    def shutdown(self) -> None:
        return None
