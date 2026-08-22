from types import SimpleNamespace

from app.agent.mcp.contracts import MCPToolCallResult


def test_from_sdk_result_normalizes_text_and_structured_content():
    result = SimpleNamespace(
        is_error=False,
        structured_content={"result": "ok"},
        content=[
            SimpleNamespace(text="first"),
            SimpleNamespace(text="second"),
            SimpleNamespace(data="ignored"),
        ],
    )

    normalized = MCPToolCallResult.from_sdk_result(result)

    assert normalized.is_error is False
    assert normalized.structured_content == {"result": "ok"}
    assert normalized.text_content == ["first", "second"]


def test_from_sdk_result_preserves_tool_error_without_structured_output():
    result = SimpleNamespace(
        is_error=True,
        structured_content=None,
        content=[SimpleNamespace(text="tool failed")],
    )

    normalized = MCPToolCallResult.from_sdk_result(result)

    assert normalized.is_error is True
    assert normalized.structured_content is None
    assert normalized.text_content == ["tool failed"]


def test_from_sdk_result_supports_mcp_v1_camel_case_fields():
    result = SimpleNamespace(
        isError=False,
        structuredContent={"result": "v2.2-release-probe"},
        content=[SimpleNamespace(text='{"result":"v2.2-release-probe"}')],
    )

    normalized = MCPToolCallResult.from_sdk_result(result)

    assert normalized.is_error is False
    assert normalized.structured_content == {"result": "v2.2-release-probe"}


def test_from_sdk_result_supports_mcp_v1_camel_case_error():
    result = SimpleNamespace(
        isError=True,
        structuredContent=None,
        content=[SimpleNamespace(text="tool failed")],
    )

    normalized = MCPToolCallResult.from_sdk_result(result)

    assert normalized.is_error is True
    assert normalized.structured_content is None
    assert normalized.text_content == ["tool failed"]
