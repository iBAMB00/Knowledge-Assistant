"""Request-level Agent trace composition for v2.5 production wiring."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from app.agent.context import ToolExecutionContext
from app.agent.observability.component import AgentComponentTracer
from app.agent.observability.context import build_agent_trace_context
from app.agent.observability.model import AgentModelTracer
from app.agent.observability.provider import AgentTraceHandle, ObservabilityProvider
from app.agent.version_snapshot import AgentRuntimeVersionSnapshot
from app.constants.agent_runtime import AgentRuntime

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class AgentRunTraceSession:
    trace_handle: AgentTraceHandle
    model_tracer: AgentModelTracer
    component_tracer: AgentComponentTracer

    def finish(self, *, ok: bool = True, error_code: str | None = None) -> None:
        try:
            self.trace_handle.finish(ok=ok, error_code=error_code)
        except Exception:
            logger.warning("Agent observability finish failed; ignoring provider error", exc_info=True)


def start_agent_run_trace(
    *,
    provider: ObservabilityProvider,
    execution_context: ToolExecutionContext,
    runtime: AgentRuntime,
    version_snapshot: AgentRuntimeVersionSnapshot,
    model_provider: str,
    model_name: str,
    prompt_id: str,
    thread_id: str | None = None,
) -> AgentRunTraceSession | None:
    if not provider.enabled:
        return None

    trace_context = build_agent_trace_context(
        execution_context=execution_context,
        runtime=runtime,
        version_snapshot=version_snapshot,
        thread_id=thread_id,
    )
    try:
        trace_handle = provider.start_trace(trace_context=trace_context, name="agent.run")
    except Exception:
        logger.warning("Agent observability start failed; continuing without trace", exc_info=True)
        return None
    return AgentRunTraceSession(
        trace_handle=trace_handle,
        model_tracer=AgentModelTracer(
            trace_context=trace_context,
            trace_handle=trace_handle,
            model_provider=model_provider,
            model_name=model_name,
            prompt_id=prompt_id,
            prompt_version=version_snapshot.prompt_version,
        ),
        component_tracer=AgentComponentTracer(
            trace_context=trace_context,
            trace_handle=trace_handle,
        ),
    )
