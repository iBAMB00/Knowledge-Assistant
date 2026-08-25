from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.agent.context import ToolExecutionContext
from app.agent.model_response import LLMToolCall
from app.agent.observability import (
    AgentComponentCallContext,
    AgentComponentResult,
    AgentComponentTracer,
    AgentGraphExecutionMode,
    AgentObservationKind,
    AgentSpanContext,
    AgentTraceContext,
    LangfuseObservabilityProvider,
)
from app.agent.observability.provider import AgentComponentCallHandle
from app.agent.tool_dispatcher import ToolDispatcher
from app.agent.tools.base import BaseAgentTool, ToolContract, ToolRiskLevel, ToolSource
from app.constants.agent_runtime import AgentRuntime
from app.constants.user_role import UserRole


@dataclass
class RecordingComponentHandle:
    span_id: str
    finishes: list[dict[str, Any]] = field(default_factory=list)

    def finish(self, *, ok: bool = True, result: AgentComponentResult | None = None, error_code: str | None = None) -> None:
        self.finishes.append({"ok": ok, "result": result, "error_code": error_code})


@dataclass
class RecordingTraceHandle:
    trace_id: str
    provider_trace_id: str | None = "provider-trace"
    component_contexts: list[AgentComponentCallContext] = field(default_factory=list)
    component_handles: list[RecordingComponentHandle] = field(default_factory=list)

    def start_component_call(self, *, call_context: AgentComponentCallContext) -> AgentComponentCallHandle:
        self.component_contexts.append(call_context)
        handle = RecordingComponentHandle(call_context.span.span_id)
        self.component_handles.append(handle)
        return handle

    def start_model_call(self, **_: Any):
        raise AssertionError("not used")

    def finish(self, **_: Any) -> None:
        return None


class SearchInput(BaseModel):
    query: str
    top_k: int | None = None


class SearchOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    result_count: int = Field(ge=0)
    items: list[str]


class FakeKnowledgeSearchTool(BaseAgentTool[SearchInput, SearchOutput]):
    name = "search_knowledge"
    version = "1.0.0"
    description = "search"
    risk_level = ToolRiskLevel.READ_ONLY
    input_model = SearchInput
    output_model = SearchOutput

    def execute(self, db, context, tool_input):
        del db, context, tool_input
        return SearchOutput(result_count=2, items=["a", "b"])

    def extract_evidence_refs(self, output: SearchOutput) -> list[str]:
        return ["doc:1", "doc:2"]


class MCPInput(BaseModel):
    value: str


class MCPOutput(BaseModel):
    ok: bool


class FakeMCPTool(BaseAgentTool[MCPInput, MCPOutput]):
    name = "mcp_demo__lookup"
    version = "mcp-v1"
    description = "mcp lookup"
    risk_level = ToolRiskLevel.READ_ONLY
    input_model = MCPInput
    output_model = MCPOutput

    def get_contract(self) -> ToolContract:
        return ToolContract(
            name=self.name,
            version=self.version,
            description=self.description,
            risk_level=self.risk_level,
            input_schema=self.input_model.model_json_schema(),
            output_schema=self.output_model.model_json_schema(),
            source=ToolSource.MCP,
            source_id="demo",
        )

    def execute(self, db, context, tool_input):
        del db, context, tool_input
        return MCPOutput(ok=True)


def _trace() -> AgentTraceContext:
    return AgentTraceContext(
        trace_id="1" * 32,
        request_id="req-a4",
        runtime=AgentRuntime.NATIVE,
        user_id=7,
        knowledge_base_id=11,
        agent_run_id=17,
        agent_version="agent-v2.5",
        prompt_id="agent.tool-calling-system",
        prompt_version="1.1.0",
        toolset_version="toolset-v2:test",
        retrieval_config_version="retrieval-v1:test",
    )


def _context() -> ToolExecutionContext:
    return ToolExecutionContext(
        user_id=7,
        role=UserRole.USER,
        knowledge_base_id=11,
        request_id="req-a4",
        agent_run_id=17,
    )


def test_component_contract_rejects_payload_fields() -> None:
    with pytest.raises(ValidationError):
        AgentComponentCallContext(
            span=AgentSpanContext(
                trace_id="trace",
                span_id="span",
                kind=AgentObservationKind.TOOL,
                name="tool.search",
            ),
            tool_name="search",
            tool_arguments={"secret": "x"},
        )


def test_component_tracer_builds_graph_resume_metadata() -> None:
    handle = RecordingTraceHandle(trace_id=_trace().trace_id)
    tracer = AgentComponentTracer(trace_context=_trace(), trace_handle=handle)

    call = tracer.start_graph_node(
        node_name="approval",
        execution_mode=AgentGraphExecutionMode.RESUME,
        turn=2,
    )
    call.finish(
        result=AgentComponentResult(next_route="tools", state_status="running")
    )

    context = handle.component_contexts[0]
    assert context.span.kind is AgentObservationKind.GRAPH_NODE
    assert context.graph_node == "approval"
    assert context.graph_execution_mode is AgentGraphExecutionMode.RESUME
    assert handle.component_handles[0].finishes[0]["result"].next_route == "tools"


def test_dispatcher_traces_tool_and_nested_knowledge_retrieval_without_payloads() -> None:
    trace_handle = RecordingTraceHandle(trace_id=_trace().trace_id)
    tracer = AgentComponentTracer(trace_context=_trace(), trace_handle=trace_handle)
    dispatcher = ToolDispatcher([FakeKnowledgeSearchTool()])

    result = dispatcher.dispatch(
        db=None,  # type: ignore[arg-type]
        context=_context(),
        tool_call=LLMToolCall(
            id="call-1",
            name="search_knowledge",
            arguments_json='{"query":"private question","top_k":3}',
        ),
        component_tracer=tracer,
        turn=1,
    )

    assert result.output["result_count"] == 2
    assert [item.span.kind for item in trace_handle.component_contexts] == [
        AgentObservationKind.TOOL,
        AgentObservationKind.RETRIEVAL,
    ]
    tool_ctx, retrieval_ctx = trace_handle.component_contexts
    assert tool_ctx.tool_name == "search_knowledge"
    assert retrieval_ctx.top_k == 3
    assert retrieval_ctx.span.parent_span_id == tool_ctx.span.span_id
    serialized = repr(trace_handle.component_contexts)
    assert "private question" not in serialized
    assert trace_handle.component_handles[1].finishes[0]["result"] == AgentComponentResult(
        result_count=2,
        evidence_count=2,
    )


def test_dispatcher_traces_mcp_as_child_component() -> None:
    trace_handle = RecordingTraceHandle(trace_id=_trace().trace_id)
    tracer = AgentComponentTracer(trace_context=_trace(), trace_handle=trace_handle)
    dispatcher = ToolDispatcher([FakeMCPTool()])

    dispatcher.dispatch(
        db=None,  # type: ignore[arg-type]
        context=_context(),
        tool_call=LLMToolCall(
            id="call-mcp",
            name="mcp_demo__lookup",
            arguments_json='{"value":"do-not-trace"}',
        ),
        component_tracer=tracer,
    )

    kinds = [item.span.kind for item in trace_handle.component_contexts]
    assert kinds == [AgentObservationKind.TOOL, AgentObservationKind.MCP]
    mcp = trace_handle.component_contexts[1]
    assert mcp.mcp_server_id == "demo"
    assert mcp.tool_name == "mcp_demo__lookup"
    assert "do-not-trace" not in repr(trace_handle.component_contexts)


@dataclass
class FakeObservation:
    id: str
    trace_id: str = "provider-trace"
    updates: list[dict[str, Any]] = field(default_factory=list)
    ended: int = 0

    def update(self, **kwargs: Any) -> None:
        self.updates.append(kwargs)

    def end(self) -> None:
        self.ended += 1


class FakeClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.observations: list[FakeObservation] = []

    def create_trace_id(self, *, seed: str | None = None) -> str:
        del seed
        return "a" * 32

    def start_observation(self, **kwargs: Any) -> FakeObservation:
        self.calls.append(kwargs)
        obs = FakeObservation(id=f"obs-{len(self.calls)}")
        self.observations.append(obs)
        return obs

    def flush(self) -> None: pass
    def shutdown(self) -> None: pass


def test_langfuse_maps_component_kinds_without_raw_io() -> None:
    client = FakeClient()
    provider = LangfuseObservabilityProvider(client=client)
    root = provider.start_trace(trace_context=_trace())
    tracer = AgentComponentTracer(trace_context=_trace(), trace_handle=root)

    tool = tracer.start_tool(
        tool_name="search_knowledge",
        tool_version="1.0.0",
        tool_source="local",
        call_id="call-1",
        turn=1,
    )
    retrieval = tracer.start_retrieval(
        parent_span_id=tool.span_id,
        top_k=5,
        turn=1,
    )
    retrieval.finish(result=AgentComponentResult(result_count=4, evidence_count=4))
    tool.finish(result=AgentComponentResult(result_count=4, evidence_count=4))

    tool_call = client.calls[1]
    component_call = client.calls[2]
    assert tool_call["as_type"] == "tool"
    assert component_call["as_type"] == "retriever"
    assert component_call["trace_context"]["parent_span_id"] == "obs-2"
    assert component_call["metadata"]["parent_internal_span_id"] == tool.span_id
    assert component_call["metadata"]["top_k"] == 5
    assert "input" not in component_call
    assert "output" not in component_call
    assert client.observations[2].updates == [
        {"metadata": {"result_count": 4, "evidence_count": 4}}
    ]
