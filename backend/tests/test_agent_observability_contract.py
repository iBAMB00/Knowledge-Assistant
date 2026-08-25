import pytest
from pydantic import ValidationError

from app.agent.context import ToolExecutionContext
from app.agent.observability import (
    AgentObservationKind,
    AgentSpanContext,
    bind_agent_run,
    bind_thread,
    build_agent_span_context,
    build_agent_trace_context,
    new_observation_id,
)
from app.agent.version_snapshot import AgentRuntimeVersionSnapshot
from app.constants.agent_runtime import AgentRuntime
from app.constants.user_role import UserRole


def _execution_context() -> ToolExecutionContext:
    return ToolExecutionContext(
        user_id=7,
        role=UserRole.USER,
        knowledge_base_id=11,
        request_id="req-observe-1",
        conversation_id=13,
    )


def _version_snapshot() -> AgentRuntimeVersionSnapshot:
    return AgentRuntimeVersionSnapshot(
        agent_version="agent-v2.4",
        prompt_version="1.1.0",
        toolset_version="toolset-v2:abc",
        retrieval_config_version="retrieval-v1:def",
    )


@pytest.mark.parametrize(
    "runtime",
    [AgentRuntime.NATIVE, AgentRuntime.LANGCHAIN, AgentRuntime.LANGGRAPH],
)
def test_trace_context_is_shared_by_all_agent_runtimes(
    runtime: AgentRuntime,
) -> None:
    trace = build_agent_trace_context(
        execution_context=_execution_context(),
        runtime=runtime,
        version_snapshot=_version_snapshot(),
        prompt_id="agent.tool-calling-system",
        trace_id="trace-fixed",
    )

    assert trace.trace_id == "trace-fixed"
    assert trace.request_id == "req-observe-1"
    assert trace.runtime is runtime
    assert trace.user_id == 7
    assert trace.knowledge_base_id == 11
    assert trace.conversation_id == 13
    assert trace.agent_run_id is None
    assert trace.agent_version == "agent-v2.4"
    assert trace.prompt_version == "1.1.0"


def test_trace_can_bind_run_and_thread_without_changing_identity() -> None:
    trace = build_agent_trace_context(
        execution_context=_execution_context(),
        runtime=AgentRuntime.LANGGRAPH,
        version_snapshot=_version_snapshot(),
        prompt_id="agent.tool-calling-system",
        trace_id="trace-1",
    )

    with_run = bind_agent_run(trace, agent_run_id=101)
    with_thread = bind_thread(with_run, thread_id=" thread-9 ")

    assert trace.agent_run_id is None
    assert trace.thread_id is None
    assert with_thread.trace_id == "trace-1"
    assert with_thread.agent_run_id == 101
    assert with_thread.thread_id == "thread-9"


def test_span_context_uses_safe_component_contract_only() -> None:
    trace = build_agent_trace_context(
        execution_context=_execution_context(),
        runtime=AgentRuntime.NATIVE,
        version_snapshot=_version_snapshot(),
        prompt_id="agent.tool-calling-system",
        trace_id="trace-safe",
    )
    span = build_agent_span_context(
        trace_context=trace,
        kind=AgentObservationKind.RETRIEVAL,
        name="knowledge_search",
        span_id="span-1",
    )

    assert span.model_dump() == {
        "trace_id": "trace-safe",
        "span_id": "span-1",
        "parent_span_id": None,
        "kind": AgentObservationKind.RETRIEVAL,
        "name": "knowledge_search",
    }

    with pytest.raises(ValidationError):
        AgentSpanContext(
            trace_id="trace-safe",
            span_id="span-unsafe",
            kind=AgentObservationKind.TOOL,
            name="search_knowledge",
            tool_arguments={"query": "secret"},
        )


def test_observation_kinds_cover_v25_trace_components() -> None:
    assert {kind.value for kind in AgentObservationKind} == {
        "agent_run",
        "model",
        "retrieval",
        "tool",
        "mcp",
        "graph_node",
    }


def test_generated_observation_ids_are_non_empty_and_unique() -> None:
    first = new_observation_id()
    second = new_observation_id()

    assert first
    assert second
    assert first != second


def test_blank_thread_id_is_rejected_before_provider_integration() -> None:
    trace = build_agent_trace_context(
        execution_context=_execution_context(),
        runtime=AgentRuntime.LANGGRAPH,
        version_snapshot=_version_snapshot(),
        prompt_id="agent.tool-calling-system",
    )

    with pytest.raises(ValueError, match="thread_id cannot be empty"):
        bind_thread(trace, thread_id="   ")
