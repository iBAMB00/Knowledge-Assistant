"""Langfuse Python SDK v4 adapter for the provider-neutral AgentOps contract."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Protocol

from app.agent.observability.contracts import (
    AgentModelCallContext,
    AgentModelUsage,
    AgentTraceContext,
)
from app.agent.observability.noop import NoOpModelCallHandle, NoOpTraceHandle
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
class LangfuseTraceHandle:
    """Small wrapper that prevents Langfuse SDK objects leaking into runtimes."""

    trace_id: str
    _provider_trace_id: str
    _observation: _LangfuseObservation
    _client: _LangfuseClient
    _finished: bool = False

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

    def finish(
        self,
        *,
        ok: bool = True,
        error_code: str | None = None,
    ) -> None:
        if self._finished:
            return

        try:
            if not ok:
                self._observation.update(
                    level="ERROR",
                    status_message=_normalize_error_code(error_code),
                )
            self._observation.end()
        except Exception:  # pragma: no cover - vendor/network defensive boundary
            logger.warning("Langfuse trace finish failed", exc_info=True)
        finally:
            self._finished = True


class LangfuseObservabilityProvider:
    """Fail-open adapter around the Langfuse v4 client.

    A2 emits safe root metadata. A3 adds model generation observations with
    model identity, prompt identity/version and token usage, but still excludes
    user prompts, model outputs, Tool payloads and retrieved document bodies.
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
            observation = self._client.start_observation(
                trace_context={"trace_id": provider_trace_id},
                name=name,
                as_type="agent",
                metadata=_safe_trace_metadata(trace_context),
                version=trace_context.agent_version,
            )
            return LangfuseTraceHandle(
                trace_id=trace_context.trace_id,
                _provider_trace_id=provider_trace_id,
                _observation=observation,
                _client=self._client,
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
