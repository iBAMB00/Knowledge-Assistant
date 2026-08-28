"""Request-level Agent trace composition for v2.5 production wiring."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from app.agent.context import ToolExecutionContext
from app.agent.observability.component import AgentComponentTracer
from app.agent.observability.context import build_agent_trace_context
from app.agent.observability.contracts import (
    AgentObservationError,
    AgentRunMetrics,
    AgentRunOutcome,
    AgentTraceContext,
)
from app.agent.observability.metrics import AgentRunMetricsCollector
from app.agent.observability.model import AgentModelTracer
from app.agent.observability.noop import NoOpTraceHandle
from app.agent.observability.pricing import AgentModelPricing
from app.agent.observability.provider import AgentTraceHandle, ObservabilityProvider
from app.agent.prompt_ops import AgentPromptReference, AgentTracePromptLink
from app.agent.version_snapshot import AgentRuntimeVersionSnapshot
from app.constants.agent_runtime import AgentRuntime

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class AgentRunTraceSession:
    trace_context: AgentTraceContext
    trace_handle: AgentTraceHandle
    model_tracer: AgentModelTracer
    component_tracer: AgentComponentTracer
    metrics_collector: AgentRunMetricsCollector

    @property
    def prompt_link(self) -> AgentTracePromptLink:
        return AgentTracePromptLink(
            trace_id=self.trace_context.trace_id,
            provider_trace_id=self.trace_handle.provider_trace_id,
            agent_run_id=self.trace_context.agent_run_id,
            runtime=self.trace_context.runtime,
            agent_version=self.trace_context.agent_version,
            prompt=AgentPromptReference(
                prompt_id=self.trace_context.prompt_id,
                prompt_version=self.trace_context.prompt_version,
            ),
        )

    def finish(
        self,
        *,
        ok: bool = True,
        error_code: str | None = None,
        error: AgentObservationError | None = None,
        outcome: AgentRunOutcome | None = None,
    ) -> AgentRunMetrics:
        metrics = self.metrics_collector.finish(
            success=ok,
            error_type=error_code if not ok else None,
            error=error,
            outcome=outcome,
        )
        try:
            if error is None:
                self.trace_handle.finish(
                    ok=ok,
                    error_code=error_code,
                    metrics=metrics,
                )
            else:
                try:
                    self.trace_handle.finish(
                        ok=ok,
                        error_code=error_code,
                        metrics=metrics,
                        error=error,
                    )
                except TypeError as exc:
                    if "unexpected keyword argument 'error'" not in str(exc):
                        raise
                    self.trace_handle.finish(
                        ok=ok,
                        error_code=error_code,
                        metrics=metrics,
                    )
        except Exception:
            logger.warning("Agent observability finish failed; ignoring provider error", exc_info=True)
        logger.info(
            "Agent run metrics: trace_id=%s success=%s outcome=%s "
            "run_latency_ms=%.3f model_calls=%d tool_calls=%d "
            "retrieval_calls=%d mcp_calls=%d failed_calls=%d warning_calls=%d tokens=%d "
            "estimated_cost_usd=%s first_error_fingerprint=%s first_warning_fingerprint=%s",
            self.trace_handle.trace_id,
            metrics.success,
            metrics.outcome.value,
            metrics.run_latency_ms,
            metrics.model_calls,
            metrics.tool_calls,
            metrics.retrieval_calls,
            metrics.mcp_calls,
            metrics.failed_calls,
            metrics.warning_calls,
            metrics.total_tokens,
            metrics.estimated_cost_usd,
            (metrics.first_error.fingerprint if metrics.first_error is not None else None),
            (metrics.first_warning.fingerprint if metrics.first_warning is not None else None),
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
    input_text: str | None = None,
) -> AgentRunTraceSession:
    trace_context = build_agent_trace_context(
        execution_context=execution_context,
        runtime=runtime,
        version_snapshot=version_snapshot,
        prompt_id=prompt_id,
        thread_id=thread_id,
        input_preview=_bounded_input_preview(input_text),
    )
    if provider.enabled:
        try:
            trace_handle = provider.start_trace(trace_context=trace_context, name="agent.chat")
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
        trace_context=trace_context,
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


def _bounded_input_preview(value: str | None, *, max_chars: int = 500) -> str | None:
    """Keep only a small in-memory preview; provider policy decides if it is exported."""

    normalized = (value or "").strip()
    if not normalized:
        return None
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max_chars - 1].rstrip() + "…"
