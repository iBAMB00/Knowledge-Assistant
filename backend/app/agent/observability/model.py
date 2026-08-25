"""Provider-neutral model-call tracing helpers for v2.5-A3."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from app.agent.observability.context import build_agent_span_context
from app.agent.observability.contracts import (
    AgentModelCallContext,
    AgentModelCallMode,
    AgentModelUsage,
    AgentObservationKind,
    AgentTraceContext,
)
from app.agent.observability.metrics import AgentRunMetricsCollector
from app.agent.observability.noop import NoOpModelCallHandle
from app.agent.observability.provider import AgentModelCallHandle, AgentTraceHandle

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class _FailOpenModelCallHandle:
    """Finish one model span and always record local run metrics exactly once."""

    delegate: AgentModelCallHandle
    metrics_collector: AgentRunMetricsCollector | None = None
    started_ns: int | None = None
    _finished: bool = False

    @property
    def span_id(self) -> str:
        return self.delegate.span_id

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
            self.delegate.finish(
                ok=ok,
                usage=usage,
                error_code=error_code,
            )
        except Exception:
            logger.warning(
                "Model observability finish failed; ignoring provider error",
                exc_info=True,
            )
        finally:
            if self.metrics_collector is not None and self.started_ns is not None:
                self.metrics_collector.record_model(
                    started_ns=self.started_ns,
                    ok=ok,
                    usage=usage,
                )
            self._finished = True


@dataclass(frozen=True, slots=True)
class AgentModelTracer:
    """Create safe model-call observations under one active Agent trace."""

    trace_context: AgentTraceContext
    trace_handle: AgentTraceHandle
    model_provider: str
    model_name: str
    prompt_id: str
    prompt_version: str
    metrics_collector: AgentRunMetricsCollector | None = None

    def start_call(
        self,
        *,
        turn: int | None = None,
        mode: AgentModelCallMode = AgentModelCallMode.TOOL_CALLING,
        name: str = "model.call",
    ) -> AgentModelCallHandle:
        span_context = build_agent_span_context(
            trace_context=self.trace_context,
            kind=AgentObservationKind.MODEL,
            name=name,
        )
        call_context = AgentModelCallContext(
            span=span_context,
            model_provider=self.model_provider,
            model_name=self.model_name,
            prompt_id=self.prompt_id,
            prompt_version=self.prompt_version,
            mode=mode,
            turn=turn,
        )
        started_ns = (
            self.metrics_collector.begin_call()
            if self.metrics_collector is not None
            else None
        )
        try:
            return _FailOpenModelCallHandle(
                self.trace_handle.start_model_call(call_context=call_context),
                metrics_collector=self.metrics_collector,
                started_ns=started_ns,
            )
        except Exception:  # defensive boundary for custom providers
            logger.warning(
                "Model observability start failed; degrading to no-op",
                exc_info=True,
            )
            no_op = NoOpModelCallHandle(span_id=span_context.span_id)
            return _FailOpenModelCallHandle(
                no_op,
                metrics_collector=self.metrics_collector,
                started_ns=started_ns,
            )


def extract_openai_usage(response: Any) -> AgentModelUsage | None:
    """Normalize OpenAI-compatible ``response.usage`` without provider coupling."""

    usage = getattr(response, "usage", None)
    if usage is None and isinstance(response, Mapping):
        usage = response.get("usage")
    return _usage_from_mapping_or_object(
        usage,
        input_names=("prompt_tokens", "input_tokens"),
        output_names=("completion_tokens", "output_tokens"),
        total_names=("total_tokens",),
    )


def extract_langchain_usage(result: Any) -> AgentModelUsage | None:
    """Read standardized ``AIMessage.usage_metadata`` from LangChain model results.

    LangChain v1 ``ModelResponse`` exposes ``result: list[BaseMessage]`` while
    ``ExtendedModelResponse`` wraps it in ``model_response``. This helper only
    reads usage metadata and never model message content.
    """

    model_response = getattr(result, "model_response", None)
    if model_response is not None:
        nested = extract_langchain_usage(model_response)
        if nested is not None:
            return nested

    messages = getattr(result, "result", None)
    if isinstance(messages, (list, tuple)):
        for message in messages:
            usage = _langchain_message_usage(message)
            if usage is not None:
                return usage

    usage = _langchain_message_usage(result)
    if usage is not None:
        return usage

    if isinstance(result, Mapping):
        raw_messages = result.get("result")
        if isinstance(raw_messages, (list, tuple)):
            for message in raw_messages:
                usage = _langchain_message_usage(message)
                if usage is not None:
                    return usage

    return None


def _langchain_message_usage(message: Any) -> AgentModelUsage | None:
    raw = (
        message.get("usage_metadata")
        if isinstance(message, Mapping)
        else getattr(message, "usage_metadata", None)
    )
    return _usage_from_mapping_or_object(
        raw,
        input_names=("input_tokens",),
        output_names=("output_tokens",),
        total_names=("total_tokens",),
    )


def _usage_from_mapping_or_object(
    raw: Any,
    *,
    input_names: tuple[str, ...],
    output_names: tuple[str, ...],
    total_names: tuple[str, ...],
) -> AgentModelUsage | None:
    if raw is None:
        return None

    input_tokens = _read_non_negative_int(raw, input_names)
    output_tokens = _read_non_negative_int(raw, output_names)
    total_tokens = _read_non_negative_int(raw, total_names)

    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens

    if input_tokens is None and output_tokens is None and total_tokens is None:
        return None

    return AgentModelUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
    )


def _read_non_negative_int(raw: Any, names: tuple[str, ...]) -> int | None:
    for name in names:
        value = raw.get(name) if isinstance(raw, Mapping) else getattr(raw, name, None)
        if isinstance(value, bool):
            continue
        if isinstance(value, int) and value >= 0:
            return value
    return None
