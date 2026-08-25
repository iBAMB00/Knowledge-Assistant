"""Langfuse Python SDK v4 adapter for the provider-neutral AgentOps contract."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Protocol

from app.agent.observability.contracts import (
    AgentComponentCallContext,
    AgentComponentResult,
    AgentModelCallContext,
    AgentModelUsage,
    AgentObservationKind,
    AgentRunMetrics,
    AgentTraceContext,
)
from app.agent.observability.noop import (
    NoOpComponentCallHandle,
    NoOpModelCallHandle,
    NoOpTraceHandle,
)
from app.core.config import Settings

logger = logging.getLogger(__name__)

_LANGFUSE_TRACE_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")


class _LangfuseObservation(Protocol):
    id: str
    trace_id: str

    def update(self, **kwargs: Any) -> Any:
        ...

    def end(self) -> Any:
        ...


class _LangfuseClient(Protocol):
    def create_trace_id(self, *, seed: str | None = None) -> str:
        ...

    def start_observation(self, **kwargs: Any) -> _LangfuseObservation:
        ...

    def flush(self) -> Any:
        ...

    def shutdown(self) -> Any:
        ...


@dataclass(slots=True)
class LangfuseModelCallHandle:
    """Generation wrapper that only emits usage/error metadata on finish."""

    span_id: str
    _observation: _LangfuseObservation
    _finished: bool = False

    def finish(
        self,
        *,
        ok: bool = True,
        usage: AgentModelUsage | None = None,
        error_code: str | None = None,
    ) -> None:
        if self._finished:
            return

        try:
            updates: dict[str, Any] = {}
            if usage is not None:
                usage_details = _langfuse_usage_details(usage)
                if usage_details:
                    updates["usage_details"] = usage_details
            if not ok:
                updates["level"] = "ERROR"
                updates["status_message"] = _normalize_error_code(error_code)
            if updates:
                self._observation.update(**updates)
            self._observation.end()
        except Exception:  # pragma: no cover - vendor/network defensive boundary
            logger.warning("Langfuse model observation finish failed", exc_info=True)
        finally:
            self._finished = True


@dataclass(slots=True)
class LangfuseComponentCallHandle:
    """Component observation wrapper with safe counters/state only."""

    span_id: str
    _observation: _LangfuseObservation
    _finished: bool = False

    def finish(
        self,
        *,
        ok: bool = True,
        result: AgentComponentResult | None = None,
        error_code: str | None = None,
    ) -> None:
        if self._finished:
            return
        try:
            updates: dict[str, Any] = {}
            if result is not None:
                metadata = _safe_component_result_metadata(result)
                if metadata:
                    updates["metadata"] = metadata
            if not ok:
                updates["level"] = "ERROR"
                updates["status_message"] = _normalize_error_code(error_code)
            if updates:
                self._observation.update(**updates)
            self._observation.end()
        except Exception:  # pragma: no cover
            logger.warning("Langfuse component observation finish failed", exc_info=True)
        finally:
            self._finished = True


@dataclass(slots=True)
class LangfuseTraceHandle:
    """Small wrapper that prevents Langfuse SDK objects leaking into runtimes."""

    trace_id: str
    _provider_trace_id: str
    _observation: _LangfuseObservation
    _client: _LangfuseClient
    _base_metadata: dict[str, str | int]
    _finished: bool = False
    _component_observation_ids: dict[str, str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self._component_observation_ids is None:
            self._component_observation_ids = {}

    @property
    def provider_trace_id(self) -> str:
        return self._provider_trace_id

    def start_model_call(
        self,
        *,
        call_context: AgentModelCallContext,
    ) -> LangfuseModelCallHandle | NoOpModelCallHandle:
        if call_context.span.trace_id != self.trace_id:
            logger.warning("Model observation trace id mismatch; degrading to no-op")
            return NoOpModelCallHandle(span_id=call_context.span.span_id)

        try:
            observation = self._client.start_observation(
                trace_context={
                    "trace_id": self._provider_trace_id,
                    "parent_span_id": self._observation.id,
                },
                name=call_context.span.name,
                as_type="generation",
                model=call_context.model_name,
                version=call_context.prompt_version,
                metadata=_safe_model_metadata(call_context),
            )
            return LangfuseModelCallHandle(
                span_id=call_context.span.span_id,
                _observation=observation,
            )
        except Exception:  # provider must never make the Agent request fail
            logger.warning(
                "Langfuse model observation start failed; degrading to no-op",
                exc_info=True,
            )
            return NoOpModelCallHandle(span_id=call_context.span.span_id)

    def start_component_call(
        self,
        *,
        call_context: AgentComponentCallContext,
    ) -> LangfuseComponentCallHandle | NoOpComponentCallHandle:
        if call_context.span.trace_id != self.trace_id:
            logger.warning("Component observation trace id mismatch; degrading to no-op")
            return NoOpComponentCallHandle(span_id=call_context.span.span_id)

        try:
            parent_provider_span_id = self._observation.id
            if call_context.span.parent_span_id is not None:
                parent_provider_span_id = self._component_observation_ids.get(
                    call_context.span.parent_span_id,
                    self._observation.id,
                )
            observation = self._client.start_observation(
                trace_context={
                    "trace_id": self._provider_trace_id,
                    "parent_span_id": parent_provider_span_id,
                },
                name=call_context.span.name,
                as_type=_component_observation_type(call_context.span.kind),
                metadata=_safe_component_metadata(call_context),
            )
            self._component_observation_ids[call_context.span.span_id] = observation.id
            return LangfuseComponentCallHandle(
                span_id=call_context.span.span_id,
                _observation=observation,
            )
        except Exception:
            logger.warning(
                "Langfuse component observation start failed; degrading to no-op",
                exc_info=True,
            )
            return NoOpComponentCallHandle(span_id=call_context.span.span_id)

    def finish(
        self,
        *,
        ok: bool = True,
        error_code: str | None = None,
        metrics: AgentRunMetrics | None = None,
    ) -> None:
        if self._finished:
            return

        try:
            updates: dict[str, Any] = {}
            if metrics is not None:
                updates["metadata"] = {
                    **self._base_metadata,
                    **_safe_run_metrics_metadata(metrics),
                }
            if not ok:
                updates["level"] = "ERROR"
                updates["status_message"] = _normalize_error_code(error_code)
            if updates:
                self._observation.update(**updates)
            self._observation.end()
        except Exception:  # pragma: no cover - vendor/network defensive boundary
            logger.warning("Langfuse trace finish failed", exc_info=True)
        finally:
            self._finished = True


class LangfuseObservabilityProvider:
    """Fail-open adapter around the Langfuse v4 client.

    A2 emits safe root metadata; A3/A4 add child observations; A5 writes a
    provider-neutral run summary (latency/counters/tokens/cost estimate) back to
    the root observation. Raw prompts, outputs and Tool/Retrieval payloads stay
    outside the observability contract.
    """

    name = "langfuse"
    enabled = True

    def __init__(self, *, client: _LangfuseClient) -> None:
        self._client = client

    @classmethod
    def from_settings(cls, settings: Settings) -> "LangfuseObservabilityProvider":
        public_key = _require_non_blank(settings.langfuse_public_key, "LANGFUSE_PUBLIC_KEY")
        secret_key = _require_non_blank(settings.langfuse_secret_key, "LANGFUSE_SECRET_KEY")

        # Lazy import keeps disabled observability from becoming a startup dependency.
        from langfuse import Langfuse

        client = Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            base_url=settings.langfuse_base_url,
            environment=settings.app_environment,
            sample_rate=settings.langfuse_sample_rate,
        )
        return cls(client=client)

    def start_trace(
        self,
        *,
        trace_context: AgentTraceContext,
        name: str = "agent.run",
    ) -> LangfuseTraceHandle | NoOpTraceHandle:
        try:
            provider_trace_id = self._resolve_provider_trace_id(trace_context.trace_id)
            base_metadata = _safe_trace_metadata(trace_context)
            observation = self._client.start_observation(
                trace_context={"trace_id": provider_trace_id},
                name=name,
                as_type="agent",
                metadata=base_metadata,
                version=trace_context.agent_version,
            )
            return LangfuseTraceHandle(
                trace_id=trace_context.trace_id,
                _provider_trace_id=provider_trace_id,
                _observation=observation,
                _client=self._client,
                _base_metadata=base_metadata,
            )
        except Exception:  # provider must never make the Agent request fail
            logger.warning("Langfuse trace start failed; degrading to no-op", exc_info=True)
            return NoOpTraceHandle(trace_id=trace_context.trace_id)

    def flush(self) -> None:
        try:
            self._client.flush()
        except Exception:  # pragma: no cover - vendor/network defensive boundary
            logger.warning("Langfuse flush failed", exc_info=True)

    def shutdown(self) -> None:
        try:
            self._client.shutdown()
        except Exception:  # pragma: no cover - vendor/network defensive boundary
            logger.warning("Langfuse shutdown failed", exc_info=True)

    def _resolve_provider_trace_id(self, trace_id: str) -> str:
        if _LANGFUSE_TRACE_ID_PATTERN.fullmatch(trace_id):
            return trace_id
        return self._client.create_trace_id(seed=trace_id)



def _safe_run_metrics_metadata(metrics: AgentRunMetrics) -> dict[str, str | int | float | bool]:
    metadata: dict[str, str | int | float | bool] = {
        "success": metrics.success,
        "run_latency_ms": metrics.run_latency_ms,
        "model_calls": metrics.model_calls,
        "tool_calls": metrics.tool_calls,
        "retrieval_calls": metrics.retrieval_calls,
        "mcp_calls": metrics.mcp_calls,
        "graph_node_calls": metrics.graph_node_calls,
        "failed_calls": metrics.failed_calls,
        "input_tokens": metrics.input_tokens,
        "output_tokens": metrics.output_tokens,
        "total_tokens": metrics.total_tokens,
        "model_usage_missing_calls": metrics.model_usage_missing_calls,
        "model_latency_ms": metrics.model_latency_ms,
        "tool_latency_ms": metrics.tool_latency_ms,
        "retrieval_latency_ms": metrics.retrieval_latency_ms,
        "mcp_latency_ms": metrics.mcp_latency_ms,
        "graph_node_latency_ms": metrics.graph_node_latency_ms,
        "pricing_configured": metrics.pricing_configured,
        "cost_estimate_complete": metrics.cost_estimate_complete,
    }
    if metrics.error_type is not None:
        metadata["error_type"] = metrics.error_type
    if metrics.estimated_cost_usd is not None:
        metadata["estimated_cost_usd"] = float(metrics.estimated_cost_usd)
    if metrics.pricing_version is not None:
        metadata["pricing_version"] = metrics.pricing_version
    return metadata

def _safe_trace_metadata(trace_context: AgentTraceContext) -> dict[str, str | int]:
    metadata: dict[str, str | int] = {
        "internal_trace_id": trace_context.trace_id,
        "request_id": trace_context.request_id,
        "runtime": trace_context.runtime.value,
        "user_id": trace_context.user_id,
        "knowledge_base_id": trace_context.knowledge_base_id,
        "agent_version": trace_context.agent_version,
        "prompt_version": trace_context.prompt_version,
        "toolset_version": trace_context.toolset_version,
        "retrieval_config_version": trace_context.retrieval_config_version,
    }
    if trace_context.conversation_id is not None:
        metadata["conversation_id"] = trace_context.conversation_id
    if trace_context.thread_id is not None:
        metadata["thread_id"] = trace_context.thread_id
    if trace_context.agent_run_id is not None:
        metadata["agent_run_id"] = str(trace_context.agent_run_id)
    return metadata


def _safe_model_metadata(call_context: AgentModelCallContext) -> dict[str, str | int]:
    metadata: dict[str, str | int] = {
        "internal_span_id": call_context.span.span_id,
        "model_provider": call_context.model_provider,
        "prompt_id": call_context.prompt_id,
        "prompt_version": call_context.prompt_version,
        "mode": call_context.mode.value,
    }
    if call_context.turn is not None:
        metadata["turn"] = call_context.turn
    return metadata


def _component_observation_type(kind: AgentObservationKind) -> str:
    if kind is AgentObservationKind.TOOL:
        return "tool"
    if kind is AgentObservationKind.RETRIEVAL:
        return "retriever"
    if kind is AgentObservationKind.MCP:
        return "tool"
    if kind is AgentObservationKind.GRAPH_NODE:
        return "chain"
    return "span"


def _safe_component_metadata(call_context: AgentComponentCallContext) -> dict[str, str | int]:
    metadata: dict[str, str | int] = {
        "internal_span_id": call_context.span.span_id,
        "kind": call_context.span.kind.value,
    }
    if call_context.span.parent_span_id is not None:
        metadata["parent_internal_span_id"] = call_context.span.parent_span_id
    for field in ("call_id", "tool_name", "tool_version", "tool_source", "mcp_server_id", "retrieval_mode", "graph_node"):
        value = getattr(call_context, field)
        if value is not None:
            metadata[field] = value
    if call_context.turn is not None:
        metadata["turn"] = call_context.turn
    if call_context.top_k is not None:
        metadata["top_k"] = call_context.top_k
    if call_context.graph_execution_mode is not None:
        metadata["graph_execution_mode"] = call_context.graph_execution_mode.value
    return metadata


def _safe_component_result_metadata(result: AgentComponentResult) -> dict[str, str | int]:
    metadata: dict[str, str | int] = {}
    if result.result_count is not None:
        metadata["result_count"] = result.result_count
    if result.evidence_count is not None:
        metadata["evidence_count"] = result.evidence_count
    if result.next_route is not None:
        metadata["next_route"] = result.next_route
    if result.state_status is not None:
        metadata["state_status"] = result.state_status
    return metadata


def _langfuse_usage_details(usage: AgentModelUsage) -> dict[str, int]:
    details: dict[str, int] = {}
    if usage.input_tokens is not None:
        details["input"] = usage.input_tokens
    if usage.output_tokens is not None:
        details["output"] = usage.output_tokens
    if usage.total_tokens is not None:
        details["total"] = usage.total_tokens
    return details


def _require_non_blank(value: str | None, env_name: str) -> str:
    normalized = (value or "").strip()
    if not normalized:
        raise ValueError(f"{env_name} is required when LANGFUSE_ENABLED=true")
    return normalized


def _normalize_error_code(error_code: str | None) -> str:
    normalized = (error_code or "agent_run_failed").strip()
    if not normalized:
        return "agent_run_failed"
    # Keep status metadata bounded and technical; never ship raw exception bodies.
    return normalized[:128]
