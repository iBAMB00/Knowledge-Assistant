import json
import sys
import types
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict, Field

from app.agent.context import ToolExecutionContext
from app.agent.frameworks.langchain.runner import (
    LangChainAgentLimitError,
    LangChainSingleAgentRunner,
)
from app.agent.run_event import (
    AgentMessageEvent,
    AgentStatusEvent,
    AgentToolCallEvent,
    AgentToolResultEvent,
)
from app.agent.tools.base import BaseAgentTool, ToolRiskLevel
from app.constants.user_role import UserRole


class DemoInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=1)


class DemoOutput(BaseModel):
    value: str


class DemoTool(BaseAgentTool[DemoInput, DemoOutput]):
    name = "demo_search"
    version = "1.0.0"
    description = "Search deterministic demo data."
    risk_level = ToolRiskLevel.READ_ONLY
    input_model = DemoInput
    output_model = DemoOutput

    def execute(self, db, context, tool_input):
        return DemoOutput(value=f"result:{tool_input.query}")


class FakeStructuredTool:
    def __init__(self, **kwargs: Any) -> None:
        self.__dict__.update(kwargs)

    @classmethod
    def from_function(cls, **kwargs: Any):
        return cls(**kwargs)


class FakeAgentMiddleware:
    """Runner 动态 Middleware 的测试父类。"""


@dataclass
class FakeAIMessage:
    content: Any
    tool_calls: list[dict[str, Any]]
    type: str = "ai"


@dataclass
class FakeToolMessage:
    content: Any
    tool_call_id: str
    name: str | None = None
    type: str = "tool"


class ClosingIterator:
    def __init__(self, chunks: list[dict[str, Any]]) -> None:
        self._iterator = iter(chunks)
        self.closed = False

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._iterator)

    def close(self) -> None:
        self.closed = True


class RecordingStreamingGraph:
    def __init__(
        self,
        *,
        chunks: list[dict[str, Any]],
        error: Exception | None = None,
    ) -> None:
        self.chunks = chunks
        self.error = error
        self.stream_input = None
        self.stream_config = None
        self.stream_mode = None
        self.iterator: ClosingIterator | None = None

    def invoke(self, input, config=None):
        raise AssertionError("stream test must not call graph.invoke")

    def stream(self, input, config=None, *, stream_mode="updates"):
        self.stream_input = input
        self.stream_config = config
        self.stream_mode = stream_mode
        if self.error is not None:
            def raise_on_next() -> Iterator[dict[str, Any]]:
                raise self.error
                yield {}
            return raise_on_next()
        self.iterator = ClosingIterator(self.chunks)
        return self.iterator


class RecordingAgentFactory:
    def __init__(self, graph: RecordingStreamingGraph) -> None:
        self.graph = graph
        self.kwargs: dict[str, Any] | None = None

    def __call__(self, **kwargs: Any):
        self.kwargs = kwargs
        return self.graph


@pytest.fixture(autouse=True)
def fake_langchain(monkeypatch):
    tools_module = types.ModuleType("langchain_core.tools")
    tools_module.StructuredTool = FakeStructuredTool
    core_package = types.ModuleType("langchain_core")
    core_package.tools = tools_module

    middleware_module = types.ModuleType("langchain.agents.middleware")
    middleware_module.AgentMiddleware = FakeAgentMiddleware
    agents_module = types.ModuleType("langchain.agents")
    agents_module.middleware = middleware_module
    langchain_package = types.ModuleType("langchain")
    langchain_package.agents = agents_module

    monkeypatch.setitem(sys.modules, "langchain_core", core_package)
    monkeypatch.setitem(sys.modules, "langchain_core.tools", tools_module)
    monkeypatch.setitem(sys.modules, "langchain", langchain_package)
    monkeypatch.setitem(sys.modules, "langchain.agents", agents_module)
    monkeypatch.setitem(
        sys.modules,
        "langchain.agents.middleware",
        middleware_module,
    )


def _context() -> ToolExecutionContext:
    return ToolExecutionContext(
        user_id=7,
        role=UserRole.USER,
        knowledge_base_id=11,
        request_id="langchain-stream-test",
        agent_run_id=99,
    )


def _runner(graph: RecordingStreamingGraph) -> LangChainSingleAgentRunner:
    return LangChainSingleAgentRunner(
        model=object(),
        tools=[DemoTool()],
        agent_factory=RecordingAgentFactory(graph),
    )


def test_langchain_runner_stream_maps_updates_to_safe_runtime_events() -> None:
    graph = RecordingStreamingGraph(
        chunks=[
            {
                "model": {
                    "messages": [
                        FakeAIMessage(
                            content=[
                                {"type": "reasoning", "text": "hidden-plan"},
                            ],
                            tool_calls=[
                                {
                                    "id": "call-1",
                                    "name": "demo_search",
                                    "args": {"query": "private-query"},
                                }
                            ],
                        )
                    ]
                }
            },
            {
                "tools": {
                    "messages": [
                        FakeToolMessage(
                            tool_call_id="call-1",
                            name="demo_search",
                            content=json.dumps(
                                {
                                    "ok": True,
                                    "result": {"secret": "private-result"},
                                }
                            ),
                        )
                    ]
                }
            },
            {
                "model": {
                    "messages": [
                        FakeAIMessage(
                            content=[
                                {"type": "reasoning", "text": "hidden-final"},
                                {"type": "text", "text": "safe final answer"},
                            ],
                            tool_calls=[],
                        )
                    ]
                }
            },
        ]
    )

    events = list(
        _runner(graph).run_events(
            db=object(),
            context=_context(),
            message="  hello  ",
        )
    )

    assert [event.type for event in events] == [
        "status",
        "tool_call",
        "tool_result",
        "status",
        "message",
    ]
    assert events[0] == AgentStatusEvent(turn=1)
    assert events[1] == AgentToolCallEvent(
        turn=1,
        call_id="call-1",
        tool_name="demo_search",
    )
    assert events[2] == AgentToolResultEvent(
        turn=1,
        call_id="call-1",
        tool_name="demo_search",
        ok=True,
        duration_ms=0,
    )
    assert events[3] == AgentStatusEvent(turn=2)
    assert events[4] == AgentMessageEvent(
        content="safe final answer",
        turns=2,
        tool_call_count=1,
    )

    rendered = "\n".join(event.model_dump_json() for event in events)
    assert "private-query" not in rendered
    assert "private-result" not in rendered
    assert "hidden-plan" not in rendered
    assert "hidden-final" not in rendered
    assert graph.stream_input == {
        "messages": [{"role": "user", "content": "hello"}]
    }
    assert graph.stream_config == {
        "recursion_limit": 32,
        "max_concurrency": 1,
    }
    assert graph.stream_mode == "updates"
    assert graph.iterator is not None and graph.iterator.closed is True


def test_langchain_runner_stream_maps_safe_tool_error_code() -> None:
    graph = RecordingStreamingGraph(
        chunks=[
            {
                "model": {
                    "messages": [
                        FakeAIMessage(
                            content="",
                            tool_calls=[
                                {
                                    "id": "call-error",
                                    "name": "demo_search",
                                    "args": {"query": "missing"},
                                }
                            ],
                        )
                    ]
                }
            },
            {
                "tools": {
                    "messages": [
                        FakeToolMessage(
                            tool_call_id="call-error",
                            name=None,
                            content=json.dumps(
                                {
                                    "ok": False,
                                    "error": {
                                        "code": "resource_not_found",
                                        "message": "sensitive internal detail",
                                    },
                                }
                            ),
                        )
                    ]
                }
            },
            {
                "model": {
                    "messages": [
                        FakeAIMessage(
                            content="没有找到资源。",
                            tool_calls=[],
                        )
                    ]
                }
            },
        ]
    )

    events = list(
        _runner(graph).run_events(
            db=object(),
            context=_context(),
            message="missing",
        )
    )
    result_event = next(
        event for event in events if isinstance(event, AgentToolResultEvent)
    )
    assert result_event.tool_name == "demo_search"
    assert result_event.ok is False
    assert result_event.error_code == "resource_not_found"
    assert "sensitive internal detail" not in result_event.model_dump_json()


def test_langchain_runner_stream_accepts_v2_updates_wrapper() -> None:
    graph = RecordingStreamingGraph(
        chunks=[
            {
                "type": "updates",
                "ns": (),
                "data": {
                    "model": {
                        "messages": [
                            FakeAIMessage(
                                content="wrapped answer",
                                tool_calls=[],
                            )
                        ]
                    }
                },
            }
        ]
    )

    events = list(
        _runner(graph).run_events(
            db=object(),
            context=_context(),
            message="hello",
        )
    )
    assert events[-1] == AgentMessageEvent(
        content="wrapped answer",
        turns=1,
        tool_call_count=0,
    )


def test_langchain_runner_stream_maps_recursion_error_and_closes_iterator() -> None:
    graph = RecordingStreamingGraph(
        chunks=[],
        error=RecursionError("framework internal recursion"),
    )
    stream = _runner(graph).run_events(
        db=object(),
        context=_context(),
        message="hello",
    )

    assert next(stream) == AgentStatusEvent(turn=1)
    with pytest.raises(LangChainAgentLimitError) as exc_info:
        next(stream)
    assert exc_info.value.code == "langchain_recursion_limit"


def test_langchain_runner_stream_close_propagates_to_graph_iterator() -> None:
    graph = RecordingStreamingGraph(
        chunks=[
            {
                "model": {
                    "messages": [
                        FakeAIMessage(
                            content="",
                            tool_calls=[
                                {
                                    "id": "call-cancel",
                                    "name": "demo_search",
                                    "args": {"query": "cancel"},
                                }
                            ],
                        )
                    ]
                }
            },
            {
                "model": {
                    "messages": [
                        FakeAIMessage(content="never reached", tool_calls=[])
                    ]
                }
            },
        ]
    )
    stream = _runner(graph).run_events(
        db=object(),
        context=_context(),
        message="cancel",
    )

    assert next(stream) == AgentStatusEvent(turn=1)
    assert isinstance(next(stream), AgentToolCallEvent)
    assert graph.iterator is not None and graph.iterator.closed is False
    stream.close()
    assert graph.iterator.closed is True
