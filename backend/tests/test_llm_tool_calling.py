import logging
from types import SimpleNamespace
from typing import Any

import pytest

import app.services.llm_service as llm_service_module
from app.agent.tools.base import ToolContract, ToolRiskLevel
from app.services.llm_service import LLMService


class FakeCompletions:
    """记录模型调用参数并返回指定响应。"""

    def __init__(self, response: Any) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return self.response


class FakeOpenAIClient:
    """LLMService 测试使用的最小 OpenAI-compatible 客户端。"""

    def __init__(self, completions: FakeCompletions) -> None:
        self.chat = SimpleNamespace(completions=completions)


def _build_service(
    monkeypatch: pytest.MonkeyPatch,
    response: Any,
) -> tuple[LLMService, FakeCompletions]:
    settings = SimpleNamespace(
        model_name="test-model",
        model_api_key="test-key",
        model_base_url="http://test-model",
    )
    completions = FakeCompletions(response=response)

    monkeypatch.setattr(
        llm_service_module,
        "get_settings",
        lambda: settings,
    )
    monkeypatch.setattr(
        llm_service_module,
        "OpenAI",
        lambda **_: FakeOpenAIClient(completions),
    )

    return LLMService(), completions


def _tool_contract(name: str = "search_knowledge") -> ToolContract:
    return ToolContract(
        name=name,
        version="1.0.0",
        description="Search the current authorized knowledge base.",
        risk_level=ToolRiskLevel.READ_ONLY,
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "top_k": {"type": "integer"},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        output_schema={
            "type": "object",
            "properties": {
                "result_count": {"type": "integer"},
            },
            "required": ["result_count"],
        },
    )


def _response(
    *,
    content: str | None = None,
    tool_calls: list[Any] | None = None,
) -> Any:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=content,
                    tool_calls=tool_calls,
                )
            )
        ]
    )


def _provider_tool_call(
    *,
    call_id: str = "call_001",
    name: str = "search_knowledge",
    arguments: str = '{"query":"Qdrant 部署"}',
) -> Any:
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(
            name=name,
            arguments=arguments,
        ),
    )


def test_chat_with_tools_sends_contract_as_model_tool_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ToolContract 必须只转换模型需要的 function schema。"""

    service, completions = _build_service(
        monkeypatch,
        response=_response(content="直接回答"),
    )
    contract = _tool_contract()

    result = service.chat_with_tools(
        "怎么部署？",
        [contract],
    )

    assert result.content == "直接回答"
    assert result.tool_calls == []

    call = completions.calls[0]
    assert call["model"] == "test-model"
    assert call["temperature"] == 0.2
    assert call["tools"] == [
        {
            "type": "function",
            "function": {
                "name": contract.name,
                "description": contract.description,
                "parameters": contract.input_schema,
            },
        }
    ]
    assert "output_schema" not in call["tools"][0]["function"]
    assert "risk_level" not in call["tools"][0]["function"]


def test_chat_with_tools_preserves_raw_tool_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """B1 只保留 Tool Call，不提前解析参数或执行 Tool。"""

    raw_arguments = '{"query":"内部部署手册","top_k":3}'
    service, _ = _build_service(
        monkeypatch,
        response=_response(
            tool_calls=[
                _provider_tool_call(
                    arguments=raw_arguments,
                )
            ]
        ),
    )

    result = service.chat_with_tools(
        "查一下部署手册",
        [_tool_contract()],
    )

    assert result.content is None
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].id == "call_001"
    assert result.tool_calls[0].name == "search_knowledge"
    assert result.tool_calls[0].arguments_json == raw_arguments


def test_chat_with_tools_rejects_empty_tool_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tool Calling 入口必须显式提供至少一个 Tool。"""

    service, completions = _build_service(
        monkeypatch,
        response=_response(content="不会执行"),
    )

    with pytest.raises(
        ValueError,
        match="tool_contracts cannot be empty",
    ):
        service.chat_with_tools("测试", [])

    assert completions.calls == []


def test_chat_with_tools_rejects_duplicate_tool_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """同一次模型调用不得暴露重名 Tool，避免 Dispatcher 歧义。"""

    service, completions = _build_service(
        monkeypatch,
        response=_response(content="不会执行"),
    )

    with pytest.raises(
        ValueError,
        match="duplicate tool name: search_knowledge",
    ):
        service.chat_with_tools(
            "测试",
            [_tool_contract(), _tool_contract()],
        )

    assert completions.calls == []


def test_chat_with_tools_rejects_empty_provider_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """模型既没有文本也没有 Tool Call 时应视为无效响应。"""

    service, _ = _build_service(
        monkeypatch,
        response=_response(),
    )

    with pytest.raises(
        RuntimeError,
        match="模型没有返回有效内容或 Tool Call",
    ):
        service.chat_with_tools(
            "测试",
            [_tool_contract()],
        )


def test_chat_with_tools_logs_do_not_contain_tool_arguments(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Tool 参数可能含企业内容，日志只记录数量而不记录参数正文。"""

    secret_arguments = '{"query":"企业机密：项目代号竹影"}'
    service, _ = _build_service(
        monkeypatch,
        response=_response(
            tool_calls=[
                _provider_tool_call(
                    arguments=secret_arguments,
                )
            ]
        ),
    )

    caplog.set_level(
        logging.INFO,
        logger="app.services.llm_service",
    )

    service.chat_with_tools(
        "查询企业资料",
        [_tool_contract()],
    )

    assert secret_arguments not in caplog.text
    assert "tool_call_count=1" in caplog.text
    assert "tool_count=1" in caplog.text
