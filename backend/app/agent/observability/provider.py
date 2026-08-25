"""Provider-neutral observability boundary for v2.5 AgentOps."""

from typing import Protocol, runtime_checkable

from app.agent.observability.contracts import AgentTraceContext


@runtime_checkable
class AgentTraceHandle(Protocol):
    """One provider-backed root Agent trace.

    Only safe lifecycle information crosses this boundary in A2. Model inputs,
    prompts, tool arguments/results and retrieved document bodies are added only
    by later explicitly reviewed instrumentation phases.
    """

    @property
    def trace_id(self) -> str:
        """Return the internal provider-neutral trace id."""
        ...

    @property
    def provider_trace_id(self) -> str | None:
        """Return the external provider trace id when one exists."""
        ...

    def finish(
        self,
        *,
        ok: bool = True,
        error_code: str | None = None,
    ) -> None:
        """Close the trace without accepting raw exception/output bodies."""
        ...


@runtime_checkable
class ObservabilityProvider(Protocol):
    """Framework-neutral tracing provider used by Native/LangChain/LangGraph."""

    @property
    def name(self) -> str:
        ...

    @property
    def enabled(self) -> bool:
        ...

    def start_trace(
        self,
        *,
        trace_context: AgentTraceContext,
        name: str = "agent.run",
    ) -> AgentTraceHandle:
        """Start one root Agent trace from trusted server-side metadata."""
        ...

    def flush(self) -> None:
        """Best-effort export of buffered observations."""
        ...

    def shutdown(self) -> None:
        """Best-effort provider shutdown."""
        ...
