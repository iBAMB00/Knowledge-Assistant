"""No-op observability implementation used when tracing is disabled/degraded."""

from dataclasses import dataclass

from app.agent.observability.contracts import AgentTraceContext


@dataclass(frozen=True, slots=True)
class NoOpTraceHandle:
    trace_id: str

    @property
    def provider_trace_id(self) -> None:
        return None

    def finish(
        self,
        *,
        ok: bool = True,
        error_code: str | None = None,
    ) -> None:
        del ok, error_code


class NoOpObservabilityProvider:
    """Provider that preserves the call contract without external side effects."""

    name = "noop"
    enabled = False

    def start_trace(
        self,
        *,
        trace_context: AgentTraceContext,
        name: str = "agent.run",
    ) -> NoOpTraceHandle:
        del name
        return NoOpTraceHandle(trace_id=trace_context.trace_id)

    def flush(self) -> None:
        return None

    def shutdown(self) -> None:
        return None
