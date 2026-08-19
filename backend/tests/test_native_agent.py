import json
from collections.abc import Sequence
from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.orm import Session

import app.agent.native_agent as native_agent_module
from app.agent.context import ToolExecutionContext
from app.agent.model_response import (
    LLMToolCall,
    LLMToolExchange,
    LLMToolResponse,
)
from app.agent.run_event import (
    AgentMessageEvent,
    AgentStatusEvent,
    AgentToolCallEvent,
    AgentToolResultEvent,
)
from app.agent.native_agent import (
    AgentRepeatedToolCallError,
    AgentTimeoutError,
    AgentToolCallLimitError,
    AgentTurnLimitError,
    NativeAgentRunner,
)
from app.agent.tools.base import (
    BaseAgentTool,
    ToolContract,
    ToolRiskLevel,
)
from app.constants.user_role import UserRole


class EchoInput(BaseModel):
    """Native Agent 测试使用的最小 Tool Input。"""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1)


class EchoOutput(BaseModel):
    """Native Agent 测试使用的最小 Tool Output。"""

    model_config = ConfigDict(extra="forbid")

    query: str
    trusted_user_id: int
    trusted_knowledge_base_id: int


class EchoTool(BaseAgentTool[EchoInput, EchoOutput]):
    """记录 Agent Runtime 实际执行次数和可信上下文。"""

    name = "echo_tool"
    version = "1.0.0"
    description = "Echo a validated query."
    risk_level = ToolRiskLevel.READ_ONLY
    input_model = EchoInput
    output_model = EchoOutput

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def execute(
        self,
        db: Session,
        context: ToolExecutionContext,
        tool_input: EchoInput,
    ) -> EchoOutput:
        self.calls.append(
            {
                "db": db,
                "context": context,
                "tool_input": tool_input,
            }
        )
        return EchoOutput(
            query=tool_input.query,
            trusted_user_id=context.user_id,
            trusted_knowledge_base_id=context.knowledge_base_id,
        )


class UnexpectedErrorTool(EchoTool):
    """模拟 Tool 内部未按 ToolError 规范处理的异常。"""

    name = "unexpected_error"

    def execute(
        self,
        db: Session,
        context: ToolExecutionContext,
        tool_input: EchoInput,
    ) -> EchoOutput:
        raise RuntimeError("provider-secret-detail")


class FakeToolCallingLLM:
    """按顺序返回预设响应，并记录每轮 provider-neutral history。"""

    def __init__(self, responses: Sequence[LLMToolResponse]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def chat_with_tool_history(
        self,
        *,
        message: str,
        tool_contracts: Sequence[ToolContract],
        history: Sequence[LLMToolExchange],
    ) -> LLMToolResponse:
        self.calls.append(
            {
                "message": message,
                "tool_contracts": list(tool_contracts),
                "history": list(history),
            }
        )

        if not self.responses:
            raise AssertionError("unexpected extra model call")

        return self.responses.pop(0)


def _context() -> ToolExecutionContext:
    return ToolExecutionContext(
        user_id=7,
        role=UserRole.USER,
        knowledge_base_id=21,
        request_id="native-agent-test-request",
    )


def _tool_call(
    *,
    call_id: str = "call_001",
    name: str = "echo_tool",
    arguments_json: str = '{"query":"部署说明"}',
) -> LLMToolCall:
    return LLMToolCall(
        id=call_id,
        name=name,
        arguments_json=arguments_json,
    )


def test_native_agent_returns_direct_answer_without_tool(
    db: Session,
) -> None:
    """模型首轮直接回答时不得无意义执行 Tool。"""

    llm = FakeToolCallingLLM(
        [LLMToolResponse(content="可以直接回答。")]
    )
    tool = EchoTool()
    runner = NativeAgentRunner(
        llm_service=llm,
        tools=[tool],
    )

    result = runner.run(
        db=db,
        context=_context(),
        message="你好",
    )

    assert result.answer == "可以直接回答。"
    assert result.turns == 1
    assert result.tool_call_count == 0
    assert tool.calls == []
    assert llm.calls[0]["history"] == []


def test_native_agent_executes_tool_and_returns_second_model_answer(
    db: Session,
) -> None:
    """B3 必须真正闭环 Model -> Tool -> Model -> Final Answer。"""

    llm = FakeToolCallingLLM(
        [
            LLMToolResponse(
                tool_calls=[_tool_call()]
            ),
            LLMToolResponse(
                content="根据工具结果，生产部署需要做好隔离。"
            ),
        ]
    )
    tool = EchoTool()
    runner = NativeAgentRunner(
        llm_service=llm,
        tools=[tool],
    )
    context = _context()

    result = runner.run(
        db=db,
        context=context,
        message=" 查询部署说明 ",
    )

    assert result.answer == "根据工具结果，生产部署需要做好隔离。"
    assert result.turns == 2
    assert result.tool_call_count == 1
    assert len(tool.calls) == 1
    assert tool.calls[0]["context"] == context
    assert tool.calls[0]["tool_input"].query == "部署说明"

    assert llm.calls[0]["message"] == "查询部署说明"
    assert len(llm.calls[1]["history"]) == 1

    exchange = llm.calls[1]["history"][0]
    payload = json.loads(exchange.tool_results[0].content_json)

    assert payload == {
        "ok": True,
        "result": {
            "query": "部署说明",
            "trusted_user_id": 7,
            "trusted_knowledge_base_id": 21,
        },
    }


def test_native_agent_returns_tool_error_to_model_without_leaking_internal_detail(
    db: Session,
) -> None:
    """ToolError 应作为结构化 observation 回填，让模型有机会恢复。"""

    llm = FakeToolCallingLLM(
        [
            LLMToolResponse(
                tool_calls=[
                    _tool_call(
                        name="unexpected_error",
                    )
                ]
            ),
            LLMToolResponse(
                content="工具暂时不可用，我无法完成该检索。"
            ),
        ]
    )
    runner = NativeAgentRunner(
        llm_service=llm,
        tools=[UnexpectedErrorTool()],
    )

    result = runner.run(
        db=db,
        context=_context(),
        message="查询资料",
    )

    payload = json.loads(
        llm.calls[1]["history"][0].tool_results[0].content_json
    )

    assert result.turns == 2
    assert payload["ok"] is False
    assert payload["error"]["code"] == "execution_failed"
    assert "provider-secret-detail" not in payload["error"]["message"]


def test_native_agent_rejects_repeated_same_tool_call(
    db: Session,
) -> None:
    """同一 Run 重复完全相同调用时必须阻止死循环和重复执行。"""

    llm = FakeToolCallingLLM(
        [
            LLMToolResponse(
                tool_calls=[
                    _tool_call(
                        call_id="call_001",
                        arguments_json='{"query":"A","x":1}',
                    )
                ]
            ),
            LLMToolResponse(
                tool_calls=[
                    _tool_call(
                        call_id="call_002",
                        arguments_json='{"x":1,"query":"A"}',
                    )
                ]
            ),
        ]
    )

    # 使用允许 x 字段的最小 Tool，避免第一轮先被 Input Schema 拒绝。
    class RepeatInput(BaseModel):
        model_config = ConfigDict(extra="forbid")
        query: str
        x: int

    class RepeatOutput(BaseModel):
        query: str

    class RepeatTool(BaseAgentTool[RepeatInput, RepeatOutput]):
        name = "echo_tool"
        version = "1.0.0"
        description = "Repeat protection test tool."
        risk_level = ToolRiskLevel.READ_ONLY
        input_model = RepeatInput
        output_model = RepeatOutput

        def __init__(self) -> None:
            self.call_count = 0

        def execute(
            self,
            db: Session,
            context: ToolExecutionContext,
            tool_input: RepeatInput,
        ) -> RepeatOutput:
            self.call_count += 1
            return RepeatOutput(query=tool_input.query)

    tool = RepeatTool()
    runner = NativeAgentRunner(
        llm_service=llm,
        tools=[tool],
    )

    with pytest.raises(
        AgentRepeatedToolCallError,
        match="repeated tool call: echo_tool",
    ):
        runner.run(
            db=db,
            context=_context(),
            message="重复调用测试",
        )

    assert tool.call_count == 1


def test_native_agent_enforces_max_tool_calls_before_execution(
    db: Session,
) -> None:
    """单轮 Tool Call 数已经超预算时不得部分执行。"""

    llm = FakeToolCallingLLM(
        [
            LLMToolResponse(
                tool_calls=[
                    _tool_call(call_id="call_001"),
                    _tool_call(
                        call_id="call_002",
                        arguments_json='{"query":"第二次"}',
                    ),
                ]
            )
        ]
    )
    tool = EchoTool()
    runner = NativeAgentRunner(
        llm_service=llm,
        tools=[tool],
        max_tool_calls=1,
    )

    with pytest.raises(
        AgentToolCallLimitError,
        match="agent exceeded max_tool_calls",
    ):
        runner.run(
            db=db,
            context=_context(),
            message="预算测试",
        )

    assert tool.calls == []


def test_native_agent_enforces_max_turns(
    db: Session,
) -> None:
    """模型持续要求新 Tool 时必须在最大轮次后终止。"""

    llm = FakeToolCallingLLM(
        [
            LLMToolResponse(
                tool_calls=[
                    _tool_call(
                        call_id="call_001",
                        arguments_json='{"query":"第一轮"}',
                    )
                ]
            ),
            LLMToolResponse(
                tool_calls=[
                    _tool_call(
                        call_id="call_002",
                        arguments_json='{"query":"第二轮"}',
                    )
                ]
            ),
        ]
    )
    tool = EchoTool()
    runner = NativeAgentRunner(
        llm_service=llm,
        tools=[tool],
        max_turns=2,
        max_tool_calls=4,
    )

    with pytest.raises(
        AgentTurnLimitError,
        match="agent exceeded max_turns",
    ):
        runner.run(
            db=db,
            context=_context(),
            message="轮次测试",
        )

    assert len(tool.calls) == 2


def test_native_agent_checks_deadline_between_operations(
    db: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """同步 Runtime 至少要在进入下一操作前阻止已超时 Run。"""

    times = iter([0.0, 0.2, 2.0])
    monkeypatch.setattr(
        native_agent_module.time,
        "monotonic",
        lambda: next(times),
    )

    llm = FakeToolCallingLLM(
        [
            LLMToolResponse(
                tool_calls=[_tool_call()]
            )
        ]
    )
    tool = EchoTool()
    runner = NativeAgentRunner(
        llm_service=llm,
        tools=[tool],
        max_duration_seconds=1.0,
    )

    with pytest.raises(
        AgentTimeoutError,
        match="agent exceeded max_duration_seconds",
    ):
        runner.run(
            db=db,
            context=_context(),
            message="超时测试",
        )

    assert tool.calls == []


def test_native_agent_rejects_invalid_runtime_limits() -> None:
    """Budget 配置必须在 Runtime 初始化阶段显式失败。"""

    llm = FakeToolCallingLLM(
        [LLMToolResponse(content="unused")]
    )
    tool = EchoTool()

    invalid_kwargs = [
        {"max_turns": 0},
        {"max_tool_calls": 0},
        {"max_duration_seconds": 0},
    ]

    for kwargs in invalid_kwargs:
        with pytest.raises(ValueError):
            NativeAgentRunner(
                llm_service=llm,
                tools=[tool],
                **kwargs,
            )


def test_native_agent_run_events_exposes_safe_tool_lifecycle(
    db: Session,
) -> None:
    """B5 Runtime 事件只暴露 Tool 生命周期，不泄漏参数或结果正文。"""

    llm = FakeToolCallingLLM(
        [
            LLMToolResponse(
                tool_calls=[
                    _tool_call(
                        arguments_json='{"query":"内部部署密钥不要外泄"}',
                    )
                ]
            ),
            LLMToolResponse(content="根据知识库结果完成回答。"),
        ]
    )
    runner = NativeAgentRunner(
        llm_service=llm,
        tools=[EchoTool()],
    )

    events = list(
        runner.run_events(
            db=db,
            context=_context(),
            message="查询内部部署资料",
        )
    )

    assert [event.type for event in events] == [
        "status",
        "tool_call",
        "tool_result",
        "status",
        "message",
    ]

    assert isinstance(events[0], AgentStatusEvent)
    assert events[0].turn == 1

    assert isinstance(events[1], AgentToolCallEvent)
    assert events[1].tool_name == "echo_tool"
    assert not hasattr(events[1], "arguments_json")

    assert isinstance(events[2], AgentToolResultEvent)
    assert events[2].ok is True
    assert events[2].error_code is None
    assert not hasattr(events[2], "content_json")

    assert isinstance(events[4], AgentMessageEvent)
    assert events[4].content == "根据知识库结果完成回答。"
    assert events[4].turns == 2
    assert events[4].tool_call_count == 1

    rendered = "\n".join(str(event.model_dump()) for event in events)
    assert "内部部署密钥不要外泄" not in rendered
    assert "trusted_user_id" not in rendered


def test_native_agent_run_events_reports_safe_tool_error_code(
    db: Session,
) -> None:
    """Tool 失败事件只暴露安全 error_code，底层异常详情仍只回填模型。"""

    llm = FakeToolCallingLLM(
        [
            LLMToolResponse(
                tool_calls=[
                    _tool_call(name="unexpected_error")
                ]
            ),
            LLMToolResponse(content="工具失败后给出降级回答。"),
        ]
    )
    runner = NativeAgentRunner(
        llm_service=llm,
        tools=[UnexpectedErrorTool()],
    )

    events = list(
        runner.run_events(
            db=db,
            context=_context(),
            message="查询资料",
        )
    )

    result_event = next(
        event
        for event in events
        if isinstance(event, AgentToolResultEvent)
    )

    assert result_event.ok is False
    assert result_event.error_code == "execution_failed"
    assert "provider-secret-detail" not in str(result_event.model_dump())
