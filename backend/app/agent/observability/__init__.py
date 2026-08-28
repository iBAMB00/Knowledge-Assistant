"""Framework-neutral Agent observability contracts and providers."""

from app.agent.observability.component import AgentComponentTracer
from app.agent.observability.contracts import (
    AgentComponentCallContext,
    AgentComponentResult,
    AgentErrorStage,
    AgentGraphExecutionMode,
    AgentModelCallContext,
    AgentModelCallMode,
    AgentModelCost,
    AgentModelUsage,
    AgentObservationError,
    AgentObservationKind,
    AgentRunMetrics,
    AgentRunOutcome,
    AgentSpanContext,
    AgentTraceContext,
)
from app.agent.observability.context import (
    bind_agent_run,
    bind_thread,
    build_agent_span_context,
    build_agent_trace_context,
    new_observation_id,
)
from app.agent.observability.error import build_control_error, build_observation_error
from app.agent.observability.factory import build_observability_provider
from app.agent.observability.langfuse_provider import LangfuseObservabilityProvider
from app.agent.observability.metrics import AgentRunMetricsCollector
from app.agent.observability.model import (
    AgentModelTracer,
    extract_langchain_usage,
    extract_openai_usage,
)
from app.agent.observability.pricing import (
    AgentModelPricing,
    build_agent_model_pricing,
    estimate_model_cost_details_usd,
    estimate_model_cost_usd,
)
from app.agent.observability.noop import (
    NoOpComponentCallHandle,
    NoOpModelCallHandle,
    NoOpObservabilityProvider,
    NoOpTraceHandle,
)
from app.agent.observability.provider import (
    AgentComponentCallHandle,
    AgentModelCallHandle,
    AgentTraceHandle,
    ObservabilityProvider,
)
from app.agent.observability.run import AgentRunTraceSession, start_agent_run_trace

__all__ = [
    "AgentComponentCallContext",
    "AgentComponentCallHandle",
    "AgentComponentResult",
    "AgentComponentTracer",
    "AgentErrorStage",
    "AgentGraphExecutionMode",
    "AgentModelCallContext",
    "AgentModelCallHandle",
    "AgentModelCallMode",
    "AgentModelCost",
    "AgentModelTracer",
    "AgentModelUsage",
    "AgentObservationError",
    "AgentModelPricing",
    "AgentObservationKind",
    "AgentRunMetrics",
    "AgentRunOutcome",
    "AgentRunMetricsCollector",
    "AgentRunTraceSession",
    "AgentSpanContext",
    "AgentTraceContext",
    "AgentTraceHandle",
    "LangfuseObservabilityProvider",
    "NoOpComponentCallHandle",
    "NoOpModelCallHandle",
    "NoOpObservabilityProvider",
    "NoOpTraceHandle",
    "ObservabilityProvider",
    "bind_agent_run",
    "bind_thread",
    "build_agent_model_pricing",
    "build_control_error",
    "build_observation_error",
    "build_agent_span_context",
    "build_agent_trace_context",
    "build_observability_provider",
    "extract_langchain_usage",
    "estimate_model_cost_details_usd",
    "estimate_model_cost_usd",
    "extract_openai_usage",
    "new_observation_id",
    "start_agent_run_trace",
]
