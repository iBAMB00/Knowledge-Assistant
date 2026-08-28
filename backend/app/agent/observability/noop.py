"""No-op observability implementation used when tracing is disabled/degraded."""

from dataclasses import dataclass

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


@dataclass(frozen=True, slots=True)
class NoOpComponentCallHandle:
    span_id: str

    def finish(self, *, ok: bool = True, result: AgentComponentResult | None = None, error_code: str | None = None, error: AgentObservationError | None = None, warning: AgentObservationError | None = None) -> None:
        del ok, result, error_code, error, warning


@dataclass(frozen=True, slots=True)
class NoOpModelCallHandle:
    span_id: str

    def finish(self, *, ok: bool = True, usage: AgentModelUsage | None = None, cost: AgentModelCost | None = None, error_code: str | None = None, error: AgentObservationError | None = None) -> None:
        del ok, usage, cost, error_code, error


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

    def finish(
        self,
        *,
        ok: bool = True,
        error_code: str | None = None,
        metrics: AgentRunMetrics | None = None,
        error: AgentObservationError | None = None,
    ) -> None:
        del ok, error_code, metrics, error


class NoOpObservabilityProvider:
    name = "noop"
    enabled = False

    def start_trace(self, *, trace_context: AgentTraceContext, name: str = "agent.chat") -> NoOpTraceHandle:
        del name
        return NoOpTraceHandle(trace_id=trace_context.trace_id)

    def publish_trace_score(self, *, provider_trace_id: str, name: str, value: float, data_type: str = "NUMERIC", comment: str | None = None, score_id: str | None = None) -> bool:
        del provider_trace_id, name, value, data_type, comment, score_id
        return False

    def flush(self) -> None:
        return None

    def shutdown(self) -> None:
        return None
