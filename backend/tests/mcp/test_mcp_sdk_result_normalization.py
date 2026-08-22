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
