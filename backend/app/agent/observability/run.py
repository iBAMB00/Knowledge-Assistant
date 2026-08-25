"""Request-level Agent trace composition for v2.5 production wiring."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from app.agent.context import ToolExecutionContext
from app.agent.observability.component import AgentComponentTracer
from app.agent.observability.context import build_agent_trace_context
from app.agent.observability.contracts import AgentRunMetrics
from app.agent.observability.metrics import AgentRunMetricsCollector
from app.agent.observability.model import AgentModelTracer
from app.agent.observability.noop import NoOpTraceHandle
from app.agent.observability.pricing import AgentModelPricing
from app.agent.observability.provider import AgentTraceHandle, ObservabilityProvider
from app.agent.version_snapshot import AgentRuntimeVersionSnapshot
from app.constants.agent_runtime import AgentRuntime

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class AgentRunTraceSession:
    trace_handle: AgentTraceHandle
    model_tracer: AgentModelTracer
    component_tracer: AgentComponentTracer
    metrics_collector: AgentRunMetricsCollector

    def finish(
        self,
        *,
        ok: bool = True,
        error_code: str | None = None,
    ) -> AgentRunMetrics:
        metrics = self.metrics_collector.finish(
            success=ok,
            error_type=error_code if not ok else None,
        )
        try:
            self.trace_handle.finish(
                ok=ok,
                error_code=error_code,
                metrics=metrics,
            )
        except Exception:
            logger.warning("Agent observability finish failed; ignoring provider error", exc_info=True)
        logger.info(
            "Agent run metrics: trace_id=%s success=%s run_latency_ms=%.3f "
            "model_calls=%d tool_calls=%d retrieval_calls=%d mcp_calls=%d "
            "tokens=%d estimated_cost_usd=%s",
            self.trace_handle.trace_id,
            metrics.success,
            metrics.run_latency_ms,
            metrics.model_calls,
            metrics.tool_calls,
            metrics.retrieval_calls,
            metrics.mcp_calls,
            metrics.total_tokens,
            metrics.estimated_cost_usd,
        )
        return metrics


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
    model_pricing: AgentModelPricing | None = None,
) -> AgentRunTraceSession:
    trace_context = build_agent_trace_context(
        execution_context=execution_context,
        runtime=runtime,
        version_snapshot=version_snapshot,
        thread_id=thread_id,
    )
    if provider.enabled:
        try:
            trace_handle = provider.start_trace(trace_context=trace_context, name="agent.run")
        except Exception:
            logger.warning(
                "Agent observability start failed; degrading external trace to no-op",
                exc_info=True,
            )
            trace_handle = NoOpTraceHandle(trace_id=trace_context.trace_id)
    else:
        trace_handle = NoOpTraceHandle(trace_id=trace_context.trace_id)

    # Local metrics are deliberately independent from the external provider.
    metrics_collector = AgentRunMetricsCollector(pricing=model_pricing)
    return AgentRunTraceSession(
        trace_handle=trace_handle,
        model_tracer=AgentModelTracer(
            trace_context=trace_context,
            trace_handle=trace_handle,
            model_provider=model_provider,
            model_name=model_name,
            prompt_id=prompt_id,
            prompt_version=version_snapshot.prompt_version,
            metrics_collector=metrics_collector,
        ),
        component_tracer=AgentComponentTracer(
            trace_context=trace_context,
            trace_handle=trace_handle,
            metrics_collector=metrics_collector,
        ),
        metrics_collector=metrics_collector,
    )
