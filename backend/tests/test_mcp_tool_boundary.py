import json
from typing import Any

import pytest
from pydantic import ValidationError

from app.agent.context import ToolExecutionContext
from app.agent.frameworks.langchain.tool_adapter import LangChainToolAdapter
from app.agent.mcp.contracts import (
    MCPRemoteToolDescriptor,
    MCPToolAnnotations,
    MCPToolCallResult,
    build_mcp_exposed_tool_name,
)
from app.agent.mcp.tool import MCPBackedAgentTool, MCPRemoteToolError
from app.agent.model_response import LLMToolCall
from app.agent.tool_dispatcher import ToolDispatcher
from app.agent.tools.base import (
    ToolExecutionError,
    ToolInvalidArgumentsError,
    ToolRiskLevel,
    ToolSource,
)
from app.constants.user_role import UserRole


class FakeMCPInvoker:
    """记录 Host 传入的远端调用参数，模拟 A2 后面的 transport。"""

    def __init__(self, result: MCPToolCallResult | None = None) -> None:
        self.result = result or MCPToolCallResult(
            structured_content={"answer": "ok"},
            text_content=["ok"],
        )
        self.calls: list[dict[str, Any]] = []

    def call_tool(
        self,
        *,
        server_id: str,
        tool_name: str,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> MCPToolCallResult:
        self.calls.append(
            {
                "server_id": server_id,
                "tool_name": tool_name,
                "arguments": arguments,
                "context": context,
            }
        )
        return self.result


class FakeStructuredTool:
    """支持 JSON Schema dict 的最小 StructuredTool 测试替身。"""

    def __init__(self, **kwargs: Any) -> None:
        self.func = kwargs["func"]
        self.name = kwargs["name"]
        self.description = kwargs["description"]
        self.args_schema = kwargs["args_schema"]
        self.infer_schema = kwargs["infer_schema"]
        self.handle_validation_error = kwargs.get("handle_validation_error")

    @classmethod
    def from_function(cls, **kwargs: Any) -> "FakeStructuredTool":
        return cls(**kwargs)

    def invoke(self, arguments: dict[str, Any]) -> str:
        return self.func(**arguments)


def _context() -> ToolExecutionContext:
    return ToolExecutionContext(
        user_id=7,
        role=UserRole.USER,
        knowledge_base_id=21,
        request_id="mcp-boundary-test",
        agent_run_id=88,
    )


def _descriptor(**overrides: Any) -> MCPRemoteToolDescriptor:
    values: dict[str, Any] = {
        "server_id": "corp_docs",
        "remote_name": "search.docs",
        "title": "Search corporate documents",
        "description": "Search a trusted corporate document service.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "minLength": 1},
                "limit": {"type": "integer", "minimum": 1, "maximum": 10},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        "output_schema": {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
            "additionalProperties": False,
        },
        "annotations": MCPToolAnnotations(
            read_only_hint=False,
            destructive_hint=True,
        ),
    }
    values.update(overrides)
    return MCPRemoteToolDescriptor(**values)


def _tool(
    *,
    descriptor: MCPRemoteToolDescriptor | None = None,
    invoker: FakeMCPInvoker | None = None,
) -> tuple[MCPBackedAgentTool, FakeMCPInvoker]:
    actual_invoker = invoker or FakeMCPInvoker()
    return (
        MCPBackedAgentTool(
            descriptor=descriptor or _descriptor(),
            invoker=actual_invoker,
            approved_risk_level=ToolRiskLevel.READ_ONLY,
        ),
        actual_invoker,
    )


def test_mcp_descriptor_rejects_model_visible_trusted_scope() -> None:
    schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "knowledge_base_id": {"type": "integer"},
        },
    }

    with pytest.raises(
        ValidationError,
        match="cannot expose trusted context fields",
    ):
        _descriptor(input_schema=schema)


def test_mcp_tool_contract_uses_local_policy_not_remote_annotations() -> None:
    tool, _ = _tool()

    contract = tool.get_contract()

    assert contract.name.startswith("mcp__corp_docs__search_docs")
    assert contract.version.startswith("mcp-v1:")
    assert contract.source is ToolSource.MCP
    assert contract.source_id == "corp_docs"
    assert contract.risk_level is ToolRiskLevel.READ_ONLY
    assert set(contract.input_schema["properties"]) == {"query", "limit"}
    assert "user_id" not in contract.input_schema["properties"]
    assert tool.descriptor.annotations.destructive_hint is True


def test_mcp_dispatcher_passes_trusted_context_out_of_band() -> None:
    tool, invoker = _tool()
    dispatcher = ToolDispatcher([tool])
    context = _context()

    result = dispatcher.dispatch(
        db=object(),  # type: ignore[arg-type]
        context=context,
        tool_call=LLMToolCall(
            id="call-mcp-1",
            name=tool.name,
            arguments_json=json.dumps({"query": "deployment", "limit": 3}),
        ),
    )

    assert result.tool_name == tool.name
    assert result.output == {
        "structured_content": {"answer": "ok"},
        "text_content": ["ok"],
    }
    assert result.evidence_refs == []
    assert len(invoker.calls) == 1
    assert invoker.calls[0]["arguments"] == {
        "query": "deployment",
        "limit": 3,
    }
    assert invoker.calls[0]["context"] is context
    assert "user_id" not in invoker.calls[0]["arguments"]
    assert "knowledge_base_id" not in invoker.calls[0]["arguments"]


def test_mcp_invalid_arguments_fail_before_remote_call() -> None:
    tool, invoker = _tool()
    dispatcher = ToolDispatcher([tool])

    with pytest.raises(ToolInvalidArgumentsError):
        dispatcher.dispatch(
            db=object(),  # type: ignore[arg-type]
            context=_context(),
            tool_call=LLMToolCall(
                id="call-mcp-invalid",
                name=tool.name,
                arguments_json=json.dumps({"query": "x", "limit": 999}),
            ),
        )

    assert invoker.calls == []


def test_mcp_remote_error_maps_to_stable_tool_error_without_remote_text() -> None:
    invoker = FakeMCPInvoker(
        MCPToolCallResult(
            is_error=True,
            text_content=["secret backend trace / token=abc"],
        )
    )
    tool, _ = _tool(invoker=invoker)
    dispatcher = ToolDispatcher([tool])

    with pytest.raises(MCPRemoteToolError) as exc_info:
        dispatcher.dispatch(
            db=object(),  # type: ignore[arg-type]
            context=_context(),
            tool_call=LLMToolCall(
                id="call-mcp-error",
                name=tool.name,
                arguments_json=json.dumps({"query": "deployment"}),
            ),
        )

    assert exc_info.value.code == "mcp_tool_error"
    assert "secret backend trace" not in str(exc_info.value)


def test_mcp_structured_output_is_validated_against_remote_contract() -> None:
    invoker = FakeMCPInvoker(
        MCPToolCallResult(structured_content={"unexpected": 1})
    )
    tool, _ = _tool(invoker=invoker)
    dispatcher = ToolDispatcher([tool])

    with pytest.raises(ToolExecutionError, match="invalid MCP structured output"):
        dispatcher.dispatch(
            db=object(),  # type: ignore[arg-type]
            context=_context(),
            tool_call=LLMToolCall(
                id="call-mcp-output",
                name=tool.name,
                arguments_json=json.dumps({"query": "deployment"}),
            ),
        )


def test_mcp_exposed_name_is_namespaced_model_safe_and_stable() -> None:
    name_a = build_mcp_exposed_tool_name(
        server_id="corp_docs",
        remote_name="search.docs",
    )
    name_b = build_mcp_exposed_tool_name(
        server_id="other_docs",
        remote_name="search.docs",
    )

    assert name_a != name_b
    assert name_a == build_mcp_exposed_tool_name(
        server_id="corp_docs",
        remote_name="search.docs",
    )
    assert len(name_a) <= 64
    assert "." not in name_a


def test_mcp_contract_version_changes_when_remote_schema_changes() -> None:
    first = _descriptor()
    changed = _descriptor(
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "minLength": 1},
                "limit": {"type": "integer", "minimum": 1, "maximum": 20},
            },
            "required": ["query"],
            "additionalProperties": False,
        }
    )

    assert first.contract_version != changed.contract_version


def test_langchain_adapter_uses_mcp_json_schema_but_still_dispatches_core_tool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool, invoker = _tool()
    adapter = LangChainToolAdapter([tool])
    monkeypatch.setattr(
        adapter,
        "_load_structured_tool",
        lambda: FakeStructuredTool,
    )

    bound = adapter.bind_tools(db=object(), context=_context())  # type: ignore[arg-type]

    assert len(bound) == 1
    assert bound[0].args_schema == tool.descriptor.input_schema

    payload = json.loads(bound[0].invoke({"query": "deployment", "limit": 2}))
    assert payload["ok"] is True
    assert payload["result"]["structured_content"] == {"answer": "ok"}
    assert len(invoker.calls) == 1


def test_mcp_declared_output_schema_requires_structured_content() -> None:
    invoker = FakeMCPInvoker(MCPToolCallResult(text_content=["text only"]))
    tool, _ = _tool(invoker=invoker)
    dispatcher = ToolDispatcher([tool])

    with pytest.raises(ToolExecutionError, match="missing MCP structured output"):
        dispatcher.dispatch(
            db=object(),  # type: ignore[arg-type]
            context=_context(),
            tool_call=LLMToolCall(
                id="call-mcp-missing-structured",
                name=tool.name,
                arguments_json=json.dumps({"query": "deployment"}),
            ),
        )
