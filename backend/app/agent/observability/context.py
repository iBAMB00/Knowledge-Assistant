"""Trace/span context creation and propagation helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import uuid4

from app.agent.context import ToolExecutionContext
from app.agent.observability.contracts import (
    AgentObservationKind,
    AgentSpanContext,
    AgentTraceContext,
)
from app.constants.agent_runtime import AgentRuntime

if TYPE_CHECKING:
    from app.agent.version_snapshot import AgentRuntimeVersionSnapshot


def new_observation_id() -> str:
    """Return a provider-neutral opaque id without external SDK dependency."""

    return uuid4().hex


def build_agent_trace_context(
    *,
    execution_context: ToolExecutionContext,
    runtime: AgentRuntime,
    version_snapshot: AgentRuntimeVersionSnapshot,
    prompt_id: str,
    trace_id: str | None = None,
    thread_id: str | None = None,
    input_preview: str | None = None,
) -> AgentTraceContext:
    """Build the immutable trace identity from trusted server-side context."""

    return AgentTraceContext(
        trace_id=trace_id or new_observation_id(),
        request_id=execution_context.request_id,
        runtime=runtime,
        user_id=execution_context.user_id,
        knowledge_base_id=execution_context.knowledge_base_id,
        conversation_id=execution_context.conversation_id,
        thread_id=thread_id,
        agent_run_id=execution_context.agent_run_id,
        input_preview=input_preview,
        agent_version=version_snapshot.agent_version,
        prompt_id=prompt_id,
        prompt_version=version_snapshot.prompt_version,
        toolset_version=version_snapshot.toolset_version,
        retrieval_config_version=version_snapshot.retrieval_config_version,
    )


def bind_agent_run(
    trace_context: AgentTraceContext,
    *,
    agent_run_id: int | str,
) -> AgentTraceContext:
    """Attach the persisted AgentRun identity without changing trace identity."""

    return trace_context.model_copy(update={"agent_run_id": agent_run_id})


def bind_thread(
    trace_context: AgentTraceContext,
    *,
    thread_id: str,
) -> AgentTraceContext:
    """Attach a Stateful Runtime thread without coupling to LangGraph types."""

    normalized = thread_id.strip()
    if not normalized:
        raise ValueError("thread_id cannot be empty")
    return trace_context.model_copy(update={"thread_id": normalized})


def build_agent_span_context(
    *,
    trace_context: AgentTraceContext,
    kind: AgentObservationKind,
    name: str,
    parent_span_id: str | None = None,
    span_id: str | None = None,
) -> AgentSpanContext:
    """Create one child span identity belonging to ``trace_context``."""

    return AgentSpanContext(
        trace_id=trace_context.trace_id,
        span_id=span_id or new_observation_id(),
        parent_span_id=parent_span_id,
        kind=kind,
        name=name,
    )
