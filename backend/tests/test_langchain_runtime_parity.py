import sys
import types
from dataclasses import dataclass
from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict

from app.agent.context import ToolExecutionContext
from app.agent.frameworks.langchain.runner import (
    LangChainAgentRepeatedToolCallError,
    LangChainAgentTimeoutError,
    LangChainAgentToolCallLimitError,
    LangChainAgentTurnLimitError,
    LangChainSingleAgentRunner,
)
from app.agent.tools.base import BaseAgentTool, ToolRiskLevel
from app.constants.user_role import UserRole
from app.services.evaluation.agent_observation_collector import (
    AgentEvaluationObservationCollector,
)


class GuardInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str


class GuardOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: str


class GuardTool(BaseAgentTool[GuardInput, GuardOutput]):
    name = "guard_tool"
    version = "1.0.0"
    description = "Deterministic runtime guard test tool."
    risk_level = ToolRiskLevel.READ_ONLY
    input_model = GuardInput
    output_model = GuardOutput

    def execute(self, db, context, tool_input):
        return GuardOutput(value=tool_input.query)


class FakeStructuredTool:
    def __init__(self, *, func, name, **_kwargs) -> None:
        self.func = func
        self.name = name

    @classmethod
    def from_function(cls, **kwargs):
        return cls(**kwargs)


class FakeAgentMiddleware:
    """LangChain AgentMiddleware 最小测试替身。"""


@dataclass
class FakeAIMessage:
    content: str
    tool_calls: list[dict[str, Any]]
    type: str = "ai"


class MiddlewareAwareGraph:
    """按 LangChain 文档的 before/after 顺序模拟最小 Agent Loop。"""

    def __init__(self, *, middleware, scripted_model_calls) -> None:
        self.middleware = list(middleware)
        self.scripted_model_calls = list(scripted_model_calls)

    def invoke(self, input, config=None):
        messages: list[Any] = list(input["messages"])

        for tool_calls in self.scripted_model_calls:
            state = {"messages": messages}
            for middleware in self.middleware:
                hook = getattr(middleware, "before_model", None)
                if hook is not None:
                    hook(state, runtime=None)

            ai_message = FakeAIMessage(content="", tool_calls=tool_calls)
            messages.append(ai_message)
            state = {"messages": messages}

            # LangChain after_* hooks reverse执行。
            for middleware in reversed(self.middleware):
                hook = getattr(middleware, "after_model", None)
                if hook is not None:
                    hook(state, runtime=None)

        # 最终回答同样来自一次模型调用，所以也要经过 before_model boundary。
        state = {"messages": messages}
        for middleware in self.middleware:
            hook = getattr(middleware, "before_model", None)
            if hook is not None:
                hook(state, runtime=None)

        messages.append(FakeAIMessage(content="safe answer", tool_calls=[]))
        state = {"messages": messages}
        for middleware in reversed(self.middleware):
            hook = getattr(middleware, "after_model", None)
            if hook is not None:
                hook(state, runtime=None)
        return {"messages": messages}


class MiddlewareAwareFactory:
    def __init__(self, scripted_model_calls) -> None:
        self.scripted_model_calls = scripted_model_calls
        self.kwargs = None

    def __call__(self, **kwargs):
        self.kwargs = kwargs
        return MiddlewareAwareGraph(
            middleware=kwargs["middleware"],
            scripted_model_calls=self.scripted_model_calls,
        )


@pytest.fixture(autouse=True)
def fake_langchain_modules(monkeypatch):
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
        request_id="req-v21-a4",
        agent_run_id=404,
    )


def _call(call_id: str, *, query: str) -> dict[str, Any]:
    return {
        "id": call_id,
        "name": "guard_tool",
        "args": {"query": query},
    }


def test_runtime_guard_rejects_model_batch_before_exceeding_tool_budget():
    factory = MiddlewareAwareFactory(
        scripted_model_calls=[[_call("call-1", query="a"), _call("call-2", query="b")]]
    )
    runner = LangChainSingleAgentRunner(
        model=object(),
        tools=[GuardTool()],
        max_tool_calls=1,
        agent_factory=factory,
    )

    with pytest.raises(LangChainAgentToolCallLimitError) as exc_info:
        runner.run(db=object(), context=_context(), message="test")

    assert exc_info.value.code == "max_tool_calls_exceeded"


def test_runtime_guard_rejects_same_tool_and_canonical_arguments_across_turns():
    factory = MiddlewareAwareFactory(
        scripted_model_calls=[
            [_call("call-1", query="same")],
            [_call("call-2", query="same")],
        ]
    )
    runner = LangChainSingleAgentRunner(
        model=object(),
        tools=[GuardTool()],
        agent_factory=factory,
    )

    with pytest.raises(LangChainAgentRepeatedToolCallError) as exc_info:
        runner.run(db=object(), context=_context(), message="test")

    assert exc_info.value.code == "repeated_tool_call"


def test_runtime_guard_preserves_observation_of_rejected_repeated_decision():
    factory = MiddlewareAwareFactory(
        scripted_model_calls=[
            [_call("call-1", query="same")],
            [_call("call-2", query="same")],
        ]
    )
    collector = AgentEvaluationObservationCollector()
    runner = LangChainSingleAgentRunner(
        model=object(),
        tools=[GuardTool()],
        agent_factory=factory,
    )

    with pytest.raises(LangChainAgentRepeatedToolCallError):
        runner.run(
            db=object(),
            context=_context(),
            message="test",
            observer=collector,
        )

    # 与 Native 一样：模型提出的错误调用本身仍进入 Eval Observation。
    calls = collector.build_tool_calls()
    assert [call.arguments for call in calls] == [
        {"query": "same"},
        {"query": "same"},
    ]


def test_runtime_guard_enforces_operation_boundary_timeout(monkeypatch):
    factory = MiddlewareAwareFactory(scripted_model_calls=[])
    runner = LangChainSingleAgentRunner(
        model=object(),
        tools=[GuardTool()],
        max_duration_seconds=1.0,
        agent_factory=factory,
    )
    times = iter([10.0, 11.0])
    monkeypatch.setattr(
        "app.agent.frameworks.langchain.runner.time.monotonic",
        lambda: next(times),
    )

    with pytest.raises(LangChainAgentTimeoutError) as exc_info:
        runner.run(db=object(), context=_context(), message="test")

    assert exc_info.value.code == "agent_timeout"


def test_runtime_guard_constructor_rejects_invalid_budget_configuration():
    with pytest.raises(ValueError, match="max_model_turns"):
        LangChainSingleAgentRunner(
            model=object(),
            tools=[GuardTool()],
            max_model_turns=0,
        )

    with pytest.raises(ValueError, match="max_tool_calls"):
        LangChainSingleAgentRunner(
            model=object(),
            tools=[GuardTool()],
            max_tool_calls=0,
        )

    with pytest.raises(ValueError, match="max_duration_seconds"):
        LangChainSingleAgentRunner(
            model=object(),
            tools=[GuardTool()],
            max_duration_seconds=0,
        )


def test_default_model_turn_budget_allows_two_tool_rounds_and_final_answer():
    factory = MiddlewareAwareFactory(
        scripted_model_calls=[
            [_call("call-1", query="first")],
            [_call("call-2", query="second")],
        ]
    )
    runner = LangChainSingleAgentRunner(
        model=object(),
        tools=[GuardTool()],
        agent_factory=factory,
    )

    result = runner.run(db=object(), context=_context(), message="test")

    assert result.answer == "safe answer"
    assert result.turns == 3
    assert result.tool_call_count == 2
    assert runner.max_model_turns == 4
    assert runner.recursion_limit == 32


def test_runtime_guard_enforces_native_equivalent_model_turn_budget():
    factory = MiddlewareAwareFactory(
        scripted_model_calls=[
            [_call("call-1", query="first")],
            [_call("call-2", query="second")],
        ]
    )
    runner = LangChainSingleAgentRunner(
        model=object(),
        tools=[GuardTool()],
        max_model_turns=2,
        agent_factory=factory,
    )

    # 两次模型轮次都产生 Tool Call 后，第 3 次（本应生成 Final Answer）
    # 在真正调用模型前被业务 max_model_turns 拦截，与 Native max_turns=2 一致。
    with pytest.raises(LangChainAgentTurnLimitError) as exc_info:
        runner.run(db=object(), context=_context(), message="test")

    assert exc_info.value.code == "max_turns_exceeded"


def test_model_turn_limit_preserves_observation_of_last_allowed_tool_decision():
    factory = MiddlewareAwareFactory(
        scripted_model_calls=[
            [_call("call-1", query="first")],
            [_call("call-2", query="second")],
        ]
    )
    collector = AgentEvaluationObservationCollector()
    runner = LangChainSingleAgentRunner(
        model=object(),
        tools=[GuardTool()],
        max_model_turns=2,
        agent_factory=factory,
    )

    with pytest.raises(LangChainAgentTurnLimitError):
        runner.run(
            db=object(),
            context=_context(),
            message="test",
            observer=collector,
        )

    assert [call.arguments for call in collector.build_tool_calls()] == [
        {"query": "first"},
        {"query": "second"},
    ]


def test_framework_recursion_limit_is_separate_generous_safety_fuse():
    default_runner = LangChainSingleAgentRunner(
        model=object(),
        tools=[GuardTool()],
    )
    larger_budget_runner = LangChainSingleAgentRunner(
        model=object(),
        tools=[GuardTool()],
        max_model_turns=6,
    )
    explicit_override_runner = LangChainSingleAgentRunner(
        model=object(),
        tools=[GuardTool()],
        max_model_turns=4,
        recursion_limit=40,
    )

    assert default_runner.max_model_turns == 4
    assert default_runner.recursion_limit == 32
    assert larger_budget_runner.recursion_limit == 48
    assert explicit_override_runner.recursion_limit == 40
