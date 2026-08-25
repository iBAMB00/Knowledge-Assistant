"""Provider-neutral observability boundary for v2.5 AgentOps."""

from typing import Protocol, runtime_checkable

from app.agent.observability.contracts import (
    AgentModelCallContext,
    AgentModelUsage,
    AgentTraceContext,
)


@runtime_checkable
class AgentModelCallHandle(Protocol):
    """One provider-backed model generation without raw prompt/output payloads."""

    @property
    def span_id(self) -> str:
        """Return the internal provider-neutral model span id."""
        ...

    def finish(
        self,
        *,
        ok: bool = True,
        usage: AgentModelUsage | None = None,
        error_code: str | None = None,
    ) -> None:
        """Close the generation with safe token/error metadata only."""
        ...


@runtime_checkable
class AgentTraceHandle(Protocol):
    """One provider-backed root Agent trace.

    Only explicitly reviewed technical metadata crosses this boundary. Prompt
    bodies, tool arguments/results, retrieved documents and model outputs are
    intentionally absent from the A3 contract.
    """

    @property
    def trace_id(self) -> str:
        """Return the internal provider-neutral trace id."""
        ...

    @property
    def provider_trace_id(self) -> str | None:
        """Return the external provider trace id when one exists."""
        ...

    def start_model_call(
        self,
        *,
        call_context: AgentModelCallContext,
    ) -> AgentModelCallHandle:
        """Start one child model generation under this root Agent trace."""
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
