import logging
from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.orm import Session

from app.agent.context import ToolExecutionContext
from app.agent.model_response import LLMToolCall
from app.agent.tool_dispatcher import ToolDispatcher
from app.agent.tools.base import (
    BaseAgentTool,
    ToolExecutionError,
    ToolInvalidArgumentsError,
    ToolNotFoundError,
    ToolRiskLevel,
)
from app.constants.user_role import UserRole


class EchoInput(BaseModel):
    """Dispatcher 测试使用的最小 Tool Input。"""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=10)


class EchoOutput(BaseModel):
    """Dispatcher 测试使用的最小 Tool Output。"""

    model_config = ConfigDict(extra="forbid")

    echoed_query: str
    trusted_user_id: int
    trusted_knowledge_base_id: int


class EchoTool(BaseAgentTool[EchoInput, EchoOutput]):
    """记录 Dispatcher 传入参数与可信上下文。"""

    name = "echo_tool"
    version = "1.0.0"
    description = "Echo validated arguments for dispatcher tests."
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
            echoed_query=tool_input.query,
            trusted_user_id=context.user_id,
            trusted_knowledge_base_id=context.knowledge_base_id,
        )


class BrokenOutputTool(EchoTool):
    """故意违反 Output Contract 的测试 Tool。"""

    name = "broken_output"

    def execute(  # type: ignore[override]
        self,
        db: Session,
        context: ToolExecutionContext,
        tool_input: EchoInput,
    ) -> Any:
        return {"unexpected": "value"}


class UnexpectedErrorTool(EchoTool):
    """模拟未按 ToolError 规范映射异常的 Tool。"""

    name = "unexpected_error"

    def execute(
        self,
        db: Session,
        context: ToolExecutionContext,
        tool_input: EchoInput,
    ) -> EchoOutput:
        raise RuntimeError("internal provider secret")


def _context() -> ToolExecutionContext:
    return ToolExecutionContext(
        user_id=7,
        role=UserRole.USER,
        knowledge_base_id=21,
        request_id="dispatcher-test-request",
    )


def _call(
    *,
    name: str = "echo_tool",
    arguments_json: str = '{"query":"部署说明","top_k":3}',
) -> LLMToolCall:
    return LLMToolCall(
        id="call_001",
        name=name,
        arguments_json=arguments_json,
    )


def test_dispatch_executes_validated_tool_with_trusted_context(
    db: Session,
) -> None:
    """合法 Tool Call 应通过 Schema 校验后使用服务端 Context 执行。"""

    tool = EchoTool()
    dispatcher = ToolDispatcher([tool])
    context = _context()

    result = dispatcher.dispatch(
        db=db,
        context=context,
        tool_call=_call(),
    )

    assert result.call_id == "call_001"
    assert result.tool_name == "echo_tool"
    assert result.output == {
        "echoed_query": "部署说明",
        "trusted_user_id": 7,
        "trusted_knowledge_base_id": 21,
    }
    assert len(tool.calls) == 1
    assert tool.calls[0]["db"] is db
    assert tool.calls[0]["context"] is context
    assert tool.calls[0]["tool_input"] == EchoInput(
        query="部署说明",
        top_k=3,
    )


def test_dispatch_rejects_unknown_tool_before_execution(
    db: Session,
) -> None:
    """模型请求未注册 Tool 时必须明确失败。"""

    tool = EchoTool()
    dispatcher = ToolDispatcher([tool])

    with pytest.raises(
        ToolNotFoundError,
        match="tool not found: delete_everything",
    ):
        dispatcher.dispatch(
            db=db,
            context=_context(),
            tool_call=_call(name="delete_everything"),
        )

    assert tool.calls == []


def test_dispatch_rejects_invalid_json_before_execution(
    db: Session,
) -> None:
    """模型 arguments 不是合法 JSON 时不得进入 Tool。"""

    tool = EchoTool()
    dispatcher = ToolDispatcher([tool])

    with pytest.raises(
        ToolInvalidArgumentsError,
        match="invalid JSON arguments for tool: echo_tool",
    ):
        dispatcher.dispatch(
            db=db,
            context=_context(),
            tool_call=_call(arguments_json='{"query":'),
        )

    assert tool.calls == []


def test_dispatch_rejects_non_object_arguments_before_execution(
    db: Session,
) -> None:
    """Tool arguments 必须是 JSON object，而不是数组或标量。"""

    tool = EchoTool()
    dispatcher = ToolDispatcher([tool])

    with pytest.raises(
        ToolInvalidArgumentsError,
        match="tool arguments must be an object: echo_tool",
    ):
        dispatcher.dispatch(
            db=db,
            context=_context(),
            tool_call=_call(arguments_json='["部署说明"]'),
        )

    assert tool.calls == []


def test_dispatch_rejects_schema_invalid_or_trusted_fields(
    db: Session,
) -> None:
    """Pydantic Input Contract 必须拒绝类型错误与可信字段注入。"""

    tool = EchoTool()
    dispatcher = ToolDispatcher([tool])

    invalid_calls = [
        _call(arguments_json='{"query":"部署","top_k":99}'),
        _call(
            arguments_json=(
                '{"query":"部署","user_id":999,'
                '"knowledge_base_id":999}'
            )
        ),
    ]

    for tool_call in invalid_calls:
        with pytest.raises(
            ToolInvalidArgumentsError,
            match="invalid arguments for tool: echo_tool",
        ):
            dispatcher.dispatch(
                db=db,
                context=_context(),
                tool_call=tool_call,
            )

    assert tool.calls == []


def test_dispatch_rejects_duplicate_or_empty_tool_set() -> None:
    """Dispatcher 自身也必须保证 Tool 名称唯一且集合非空。"""

    with pytest.raises(ValueError, match="tools cannot be empty"):
        ToolDispatcher([])

    with pytest.raises(
        ValueError,
        match="duplicate tool name: echo_tool",
    ):
        ToolDispatcher([EchoTool(), EchoTool()])


def test_dispatch_validates_tool_output_contract(
    db: Session,
) -> None:
    """Tool 实现违反 Output Schema 时不得把脏结果传给 Agent。"""

    dispatcher = ToolDispatcher([BrokenOutputTool()])

    with pytest.raises(
        ToolExecutionError,
        match="invalid tool output: broken_output",
    ):
        dispatcher.dispatch(
            db=db,
            context=_context(),
            tool_call=_call(name="broken_output"),
        )


def test_dispatch_maps_unexpected_tool_error_without_leaking_detail(
    db: Session,
) -> None:
    """未映射的底层异常必须收口，不能把内部详情泄给 Agent。"""

    dispatcher = ToolDispatcher([UnexpectedErrorTool()])

    with pytest.raises(
        ToolExecutionError,
        match="tool execution failed: unexpected_error",
    ) as error:
        dispatcher.dispatch(
            db=db,
            context=_context(),
            tool_call=_call(name="unexpected_error"),
        )

    assert "internal provider secret" not in str(error.value)


def test_dispatch_logs_do_not_contain_arguments_or_output(
    db: Session,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Dispatcher 日志只记录调用元数据，不记录企业参数和 Tool 输出正文。"""

    secret = "企业机密：项目代号竹影"
    dispatcher = ToolDispatcher([EchoTool()])

    caplog.set_level(
        logging.INFO,
        logger="app.agent.tool_dispatcher",
    )

    dispatcher.dispatch(
        db=db,
        context=_context(),
        tool_call=_call(
            arguments_json=(
                '{"query":"' + secret + '","top_k":3}'
            )
        ),
    )

    assert secret not in caplog.text
    assert "tool_name=echo_tool" in caplog.text
    assert "call_id=call_001" in caplog.text
