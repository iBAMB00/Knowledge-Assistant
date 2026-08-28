"""Langfuse Python SDK v4 adapter for the provider-neutral AgentOps contract."""

from __future__ import annotations

import logging
import re
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Callable, ContextManager, Protocol

from app.agent.observability.contracts import (
    AgentComponentCallContext,
    AgentComponentResult,
    AgentModelCallContext,
    AgentModelCost,
    AgentModelUsage,
    AgentObservationError,
    AgentObservationKind,
    AgentRunMetrics,
    AgentRunOutcome,
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

    def update(self, **kwargs: Any) -> Any: ...

    def end(self) -> Any: ...


class _LangfuseClient(Protocol):
    def create_trace_id(self, *, seed: str | None = None) -> str: ...

    def start_observation(self, **kwargs: Any) -> _LangfuseObservation: ...

    def create_score(self, **kwargs: Any) -> Any: ...

    def flush(self) -> Any: ...

    def shutdown(self) -> Any: ...


CorrelationScopeFactory = Callable[..., ContextManager[Any]]


@dataclass(slots=True)
class LangfuseModelCallHandle:
    """Generation wrapper that emits safe usage/error metadata on finish."""

    span_id: str
    _observation: _LangfuseObservation
    _capture_safe_error_message: bool = False
    _finished: bool = False

    def finish(
        self,
        *,
        ok: bool = True,
        usage: AgentModelUsage | None = None,
        cost: AgentModelCost | None = None,
        error_code: str | None = None,
        error: AgentObservationError | None = None,
    ) -> None:
        if self._finished:
            return

        try:
            updates: dict[str, Any] = {}
            if usage is not None:
                usage_details = _langfuse_usage_details(usage)
                if usage_details:
                    updates["usage_details"] = usage_details
            if cost is not None:
                cost_details = _langfuse_cost_details(cost)
                if cost_details:
                    updates["cost_details"] = cost_details
            if not ok:
                updates["level"] = "ERROR"
                updates["status_message"] = _error_status_message(
                    error=error,
                    fallback_code=error_code,
                    include_message=self._capture_safe_error_message,
                )
                if error is not None:
                    updates["metadata"] = _safe_error_metadata(
                        error,
                        include_message=self._capture_safe_error_message,
                    )
            if updates:
                self._observation.update(**updates)
            self._observation.end()
        except Exception:  # pragma: no cover - vendor/network defensive boundary
            logger.warning("Langfuse model observation finish failed", exc_info=True)
        finally:
            self._finished = True


@dataclass(slots=True)
class LangfuseComponentCallHandle:
    """Component observation wrapper with safe counters/state/error facts only."""

    span_id: str
    _observation: _LangfuseObservation
    _capture_safe_error_message: bool = False
    _finished: bool = False

    def finish(
        self,
        *,
        ok: bool = True,
        result: AgentComponentResult | None = None,
        error_code: str | None = None,
        error: AgentObservationError | None = None,
        warning: AgentObservationError | None = None,
    ) -> None:
        if self._finished:
            return
        try:
            updates: dict[str, Any] = {}
            metadata: dict[str, str | int | bool] = {}
            if result is not None:
                metadata.update(_safe_component_result_metadata(result))
            if not ok:
                updates["level"] = "ERROR"
                updates["status_message"] = _error_status_message(
                    error=error,
                    fallback_code=error_code,
                    include_message=self._capture_safe_error_message,
                )
                if error is not None:
                    metadata.update(
                        _safe_error_metadata(
                            error,
                            include_message=self._capture_safe_error_message,
                        )
                    )
            elif warning is not None:
                updates["level"] = "WARNING"
                updates["status_message"] = _error_status_message(
                    error=warning,
                    fallback_code=warning.error_code,
                    include_message=self._capture_safe_error_message,
                )
                metadata.update(
                    _safe_error_metadata(
                        warning,
                        include_message=self._capture_safe_error_message,
                        prefix="warning_",
                    )
                )
            if metadata:
                updates["metadata"] = metadata
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
    _correlation_scope_factory: CorrelationScopeFactory
    _correlation_attributes: dict[str, str]
    _capture_safe_error_message: bool = False
    _finished: bool = False
    _component_observation_ids: dict[str, str] | None = None

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
            with self._correlation_scope_factory(**self._correlation_attributes):
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
                _capture_safe_error_message=self._capture_safe_error_message,
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
                parent_provider_span_id = (self._component_observation_ids or {}).get(
                    call_context.span.parent_span_id,
                    self._observation.id,
                )
            with self._correlation_scope_factory(**self._correlation_attributes):
                observation = self._client.start_observation(
                    trace_context={
                        "trace_id": self._provider_trace_id,
                        "parent_span_id": parent_provider_span_id,
                    },
                    name=call_context.span.name,
                    as_type=_component_observation_type(call_context.span.kind),
                    metadata=_safe_component_metadata(call_context),
                )
            if self._component_observation_ids is not None:
                self._component_observation_ids[call_context.span.span_id] = observation.id
            return LangfuseComponentCallHandle(
                span_id=call_context.span.span_id,
                _observation=observation,
                _capture_safe_error_message=self._capture_safe_error_message,
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
        error: AgentObservationError | None = None,
    ) -> None:
        if self._finished:
            return

        try:
            updates: dict[str, Any] = {}
            metadata: dict[str, str | int | float | bool] = {}
            if metrics is not None:
                metadata.update(self._base_metadata)
                metadata.update(_safe_run_metrics_metadata(metrics))
            if error is not None:
                metadata.update(
                    _safe_error_metadata(
                        error,
                        include_message=self._capture_safe_error_message,
                        prefix="run_error_",
                    )
                )
            if metadata:
                updates["metadata"] = metadata

            if metrics is not None:
                if metrics.outcome is AgentRunOutcome.FAILED:
                    updates["level"] = "ERROR"
                    updates["status_message"] = _error_status_message(
                        error=error,
                        fallback_code=error_code,
                        include_message=self._capture_safe_error_message,
                    )
                elif metrics.outcome is AgentRunOutcome.DEGRADED:
                    updates["level"] = "WARNING"
                    updates["status_message"] = (
                        "degraded: "
                        f"failed={metrics.failed_calls}, warnings={metrics.warning_calls}"
                    )
                elif metrics.outcome is AgentRunOutcome.WAITING:
                    updates["level"] = "WARNING"
                    updates["status_message"] = error_code or "waiting_for_approval"
                elif metrics.outcome is AgentRunOutcome.CANCELLED:
                    updates["level"] = "WARNING"
                    updates["status_message"] = error_code or "agent_cancelled"
            elif not ok:
                updates["level"] = "ERROR"
                updates["status_message"] = _error_status_message(
                    error=error,
                    fallback_code=error_code,
                    include_message=self._capture_safe_error_message,
                )

            self._observation.update(**updates)
            self._observation.end()
        except Exception:  # pragma: no cover - vendor/network defensive boundary
            logger.warning("Langfuse trace finish failed", exc_info=True)
        finally:
            self._finished = True


class LangfuseObservabilityProvider:
    """Fail-open adapter around the Langfuse v4 client.

    A8 adds native Langfuse user/session correlation, optional bounded request
    preview, structured safe errors and trace-score publishing while keeping raw
    prompts, Tool/Retrieval payloads and tracebacks outside the vendor boundary.
    """

    name = "langfuse"
    enabled = True

    def __init__(
        self,
        *,
        client: _LangfuseClient,
        correlation_scope_factory: CorrelationScopeFactory | None = None,
        capture_input_preview: bool = False,
        input_preview_max_chars: int = 120,
        capture_safe_error_message: bool = False,
    ) -> None:
        self._client = client
        self._correlation_scope_factory = (
            correlation_scope_factory or _null_correlation_scope_factory
        )
        self._capture_input_preview = capture_input_preview
        self._input_preview_max_chars = max(20, min(500, input_preview_max_chars))
        self._capture_safe_error_message = capture_safe_error_message

    @classmethod
    def from_settings(cls, settings: Settings) -> "LangfuseObservabilityProvider":
        public_key = _require_non_blank(settings.langfuse_public_key, "LANGFUSE_PUBLIC_KEY")
        secret_key = _require_non_blank(settings.langfuse_secret_key, "LANGFUSE_SECRET_KEY")

        # Lazy import keeps disabled observability from becoming a startup dependency.
        from langfuse import Langfuse, propagate_attributes

        client = Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            base_url=settings.langfuse_base_url,
            environment=settings.app_environment,
            sample_rate=settings.langfuse_sample_rate,
        )
        return cls(
            client=client,
            correlation_scope_factory=propagate_attributes,
            capture_input_preview=settings.langfuse_capture_input_preview,
            input_preview_max_chars=settings.langfuse_input_preview_max_chars,
            capture_safe_error_message=settings.langfuse_capture_safe_error_message,
        )

    def start_trace(
        self,
        *,
        trace_context: AgentTraceContext,
        name: str = "agent.chat",
    ) -> LangfuseTraceHandle | NoOpTraceHandle:
        try:
            provider_trace_id = self._resolve_provider_trace_id(trace_context.trace_id)
            base_metadata = _safe_trace_metadata(trace_context)
            correlation_attributes = _correlation_attributes(
                trace_context,
                trace_name=name,
            )

            start_kwargs: dict[str, Any] = {
                "trace_context": {"trace_id": provider_trace_id},
                "name": name,
                "as_type": "agent",
                "metadata": base_metadata,
                "version": trace_context.agent_version,
            }
            preview = self._exportable_input_preview(trace_context.input_preview)
            if preview is not None:
                start_kwargs["input"] = {"question_preview": preview}

            with self._correlation_scope_factory(**correlation_attributes):
                observation = self._client.start_observation(**start_kwargs)
            return LangfuseTraceHandle(
                trace_id=trace_context.trace_id,
                _provider_trace_id=provider_trace_id,
                _observation=observation,
                _client=self._client,
                _base_metadata=base_metadata,
                _correlation_scope_factory=self._correlation_scope_factory,
                _correlation_attributes=correlation_attributes,
                _capture_safe_error_message=self._capture_safe_error_message,
            )
        except Exception:  # provider must never make the Agent request fail
            logger.warning("Langfuse trace start failed; degrading to no-op", exc_info=True)
            return NoOpTraceHandle(trace_id=trace_context.trace_id)

    def publish_trace_score(
        self,
        *,
        provider_trace_id: str,
        name: str,
        value: float,
        data_type: str = "NUMERIC",
        comment: str | None = None,
        score_id: str | None = None,
    ) -> bool:
        """Publish one bounded Eval score to an existing Langfuse trace."""

        normalized_trace_id = provider_trace_id.strip()
        normalized_name = name.strip()
        if not normalized_trace_id or not normalized_name:
            return False
        try:
            kwargs: dict[str, Any] = {
                "trace_id": normalized_trace_id,
                "name": normalized_name[:100],
                "value": float(value),
                "data_type": data_type,
            }
            if comment:
                kwargs["comment"] = " ".join(comment.split())[:300]
            normalized_score_id = (score_id or "").strip()
            if normalized_score_id:
                kwargs["score_id"] = normalized_score_id[:64]
            self._client.create_score(**kwargs)
            return True
        except Exception:
            logger.warning(
                "Langfuse score publish failed; evaluation result is kept locally",
                exc_info=True,
            )
            return False

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

    def _exportable_input_preview(self, value: str | None) -> str | None:
        if not self._capture_input_preview:
            return None
        normalized = (value or "").strip()
        if not normalized:
            return None
        if len(normalized) <= self._input_preview_max_chars:
            return normalized
        return normalized[: self._input_preview_max_chars - 1].rstrip() + "…"


def _safe_run_metrics_metadata(metrics: AgentRunMetrics) -> dict[str, str | int | float | bool]:
    metadata: dict[str, str | int | float | bool] = {
        "success": metrics.success,
        "run_outcome": metrics.outcome.value,
        "run_latency_ms": metrics.run_latency_ms,
        "model_calls": metrics.model_calls,
        "tool_calls": metrics.tool_calls,
        "retrieval_calls": metrics.retrieval_calls,
        "mcp_calls": metrics.mcp_calls,
        "graph_node_calls": metrics.graph_node_calls,
        "failed_calls": metrics.failed_calls,
        "warning_calls": metrics.warning_calls,
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
    if metrics.first_error is not None:
        metadata.update(
            _safe_error_metadata(
                metrics.first_error,
                include_message=False,
                prefix="first_error_",
            )
        )
    if metrics.first_warning is not None:
        metadata.update(
            _safe_error_metadata(
                metrics.first_warning,
                include_message=False,
                prefix="first_warning_",
            )
        )
    return metadata


def _safe_trace_metadata(trace_context: AgentTraceContext) -> dict[str, str | int]:
    metadata: dict[str, str | int] = {
        "internal_trace_id": trace_context.trace_id,
        "request_id": trace_context.request_id,
        "runtime": trace_context.runtime.value,
        "user_id": trace_context.user_id,
        "knowledge_base_id": trace_context.knowledge_base_id,
        "agent_version": trace_context.agent_version,
        "prompt_id": trace_context.prompt_id,
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


def _correlation_attributes(
    trace_context: AgentTraceContext,
    *,
    trace_name: str,
) -> dict[str, str]:
    attributes = {
        "trace_name": trace_name,
        "user_id": str(trace_context.user_id),
    }
    if trace_context.conversation_id is not None:
        attributes["session_id"] = f"conversation:{trace_context.conversation_id}"
    return attributes


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
    for field in (
        "call_id",
        "tool_name",
        "tool_version",
        "tool_source",
        "mcp_server_id",
        "retrieval_mode",
        "graph_node",
    ):
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


def _safe_error_metadata(
    error: AgentObservationError,
    *,
    include_message: bool,
    prefix: str = "error_",
) -> dict[str, str | int | bool]:
    metadata: dict[str, str | int | bool] = {
        f"{prefix}code": error.error_code,
        f"{prefix}type": error.error_type,
        f"{prefix}stage": error.stage.value,
        f"{prefix}fingerprint": error.fingerprint,
    }
    optional = {
        "substage": error.substage,
        "root_cause_type": error.root_cause_type,
        "provider": error.provider,
        "model": error.model,
        "provider_error_code": error.provider_error_code,
        "http_status": error.http_status,
        "retryable": error.retryable,
        "fail_open": error.fail_open,
    }
    for key, value in optional.items():
        if value is not None:
            metadata[f"{prefix}{key}"] = value
    if include_message and error.safe_message is not None:
        metadata[f"{prefix}safe_message"] = error.safe_message
    return metadata


def _error_status_message(
    *,
    error: AgentObservationError | None,
    fallback_code: str | None,
    include_message: bool,
) -> str:
    if error is not None:
        if include_message and error.safe_message:
            return f"{error.error_code}: {error.safe_message}"[:320]
        return error.error_code[:128]
    return _normalize_error_code(fallback_code)


def _langfuse_cost_details(cost: AgentModelCost) -> dict[str, float]:
    details: dict[str, float] = {}
    if cost.input_cost_usd is not None:
        details["input"] = float(cost.input_cost_usd)
    if cost.output_cost_usd is not None:
        details["output"] = float(cost.output_cost_usd)
    if cost.total_cost_usd is not None:
        details["total"] = float(cost.total_cost_usd)
    return details


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
    return normalized[:128]


def _null_correlation_scope_factory(**_: Any) -> ContextManager[Any]:
    return nullcontext()
